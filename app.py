import os
import io
from datetime import date, timedelta
from functools import lru_cache

import pandas as pd
import requests
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

TAX_RATE = 0.19
NBP_RANGE_API = "https://api.nbp.pl/api/exchangerates/rates/a/eur/{start}/{end}/?format=json"
NBP_SINGLE_API = "https://api.nbp.pl/api/exchangerates/rates/a/eur/{date}/?format=json"


@lru_cache(maxsize=512)
def _fetch_single_rate(date_str: str):
    try:
        r = requests.get(NBP_SINGLE_API.format(date=date_str), timeout=8)
        if r.status_code == 200:
            return r.json()["rates"][0]["mid"]
    except Exception:
        pass
    return None


def get_nbp_rates_bulk(start: date, end: date) -> dict:
    url = NBP_RANGE_API.format(start=start.isoformat(), end=end.isoformat())
    try:
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            return {row["effectiveDate"]: row["mid"] for row in r.json()["rates"]}
    except Exception:
        pass
    return {}


def prev_business_day_rate(tx_date: date, rate_map: dict):
    """Kurs NBP z poprzedniego dnia roboczego (D-1)."""
    for delta in range(1, 12):
        d = tx_date - timedelta(days=delta)
        rate = rate_map.get(d.isoformat())
        if rate is not None:
            return rate, d.isoformat()
    # fallback: pojedyncze zapytania
    for delta in range(1, 12):
        d = tx_date - timedelta(days=delta)
        rate = _fetch_single_rate(d.isoformat())
        if rate is not None:
            return rate, d.isoformat()
    return None, None


def parse_money(val) -> float | None:
    """Parsuje wartości w formacie '+€2,29' lub '+€45,000'."""
    if pd.isna(val) or val is None:
        return None
    s = str(val).replace("+", "").replace("€", "").replace(",", "").strip()
    try:
        return float(s)
    except ValueError:
        return None


def parse_revolut_csv(file_bytes: bytes) -> pd.DataFrame:
    """
    Parsuje wyciąg Revolut z lokaty.
    Oczekiwane kolumny: Completed Date, Description, Money in, Money out, Balance, Product name
    """
    df = None
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            df = pd.read_csv(io.BytesIO(file_bytes), encoding=enc)
            break
        except Exception:
            continue
    if df is None:
        raise ValueError("Nie można odczytać pliku CSV.")

    df.columns = [c.strip() for c in df.columns]

    # Sprawdź czy to właściwy format Revolut savings
    required = {"Completed Date", "Description", "Money in"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Nieoczekiwany format CSV. Brakuje kolumn: {missing}. "
            f"Dostępne kolumny: {list(df.columns)}"
        )

    # Filtruj tylko odsetki (Gross interest)
    interest_df = df[df["Description"].str.contains("Gross interest", na=False, case=False)].copy()

    if interest_df.empty:
        raise ValueError(
            "Nie znaleziono wierszy z odsetkami ('Gross interest') w pliku. "
            "Upewnij się, że to wyciąg z konta oszczędnościowego Revolut."
        )

    # Parsuj datę: "17 Jun 2025"
    interest_df["date"] = pd.to_datetime(
        interest_df["Completed Date"].str.strip(), format="%d %b %Y", errors="coerce"
    ).dt.date

    # Parsuj kwotę odsetek
    interest_df["amount_eur"] = interest_df["Money in"].apply(parse_money)

    # Usuń wiersze z błędami parsowania
    bad = interest_df[interest_df["date"].isna() | interest_df["amount_eur"].isna()]
    if not bad.empty:
        print(f"[warn] Pominięto {len(bad)} wierszy z błędem parsowania")
    interest_df = interest_df.dropna(subset=["date", "amount_eur"])
    interest_df = interest_df[interest_df["amount_eur"] > 0]

    if interest_df.empty:
        raise ValueError("Po filtrowaniu nie zostały żadne prawidłowe wiersze z odsetkami.")

    return interest_df[["date", "amount_eur", "Description"]].reset_index(drop=True)


def calculate_tax(interest_df: pd.DataFrame) -> dict:
    min_date = interest_df["date"].min()
    max_date = interest_df["date"].max()

    # Pobierz kursy NBP z zapasem 15 dni (cofamy się dla weekendów/świąt)
    rate_map = get_nbp_rates_bulk(min_date - timedelta(days=15), max_date)

    records = []
    missing_rates = []

    for _, row in interest_df.iterrows():
        tx_date = row["date"]
        amount_eur = float(row["amount_eur"])

        nbp_rate, rate_date = prev_business_day_rate(tx_date, rate_map)
        if nbp_rate is None:
            missing_rates.append(str(tx_date))
            continue

        amount_pln = amount_eur * nbp_rate
        tax_pln = amount_pln * TAX_RATE

        records.append({
            "date": tx_date.isoformat(),
            "month": tx_date.strftime("%Y-%m"),
            "amount_eur": round(amount_eur, 6),
            "nbp_rate": round(nbp_rate, 4),
            "nbp_rate_date": rate_date,
            "amount_pln": round(amount_pln, 4),
            "tax_pln": round(tax_pln, 4),
        })

    if not records:
        raise ValueError("Nie udało się pobrać kursów NBP dla żadnej transakcji. Sprawdź połączenie z internetem.")

    result_df = pd.DataFrame(records)

    daily = (
        result_df.groupby("date")
        .agg(
            amount_eur=("amount_eur", "sum"),
            amount_pln=("amount_pln", "sum"),
            tax_pln=("tax_pln", "sum"),
            nbp_rate=("nbp_rate", "first"),
            nbp_rate_date=("nbp_rate_date", "first"),
        )
        .reset_index()
        .sort_values("date")
    )

    monthly = (
        result_df.groupby("month")
        .agg(
            amount_eur=("amount_eur", "sum"),
            amount_pln=("amount_pln", "sum"),
            tax_pln=("tax_pln", "sum"),
        )
        .reset_index()
        .sort_values("month")
    )

    yearly = {
        "amount_eur": round(float(result_df["amount_eur"].sum()), 4),
        "amount_pln": round(float(result_df["amount_pln"].sum()), 4),
        "tax_pln": round(float(result_df["tax_pln"].sum()), 4),
    }

    return {
        "daily": daily.round(4).to_dict(orient="records"),
        "monthly": monthly.round(4).to_dict(orient="records"),
        "yearly": yearly,
        "missing_rates": missing_rates,
        "tax_rate_pct": TAX_RATE * 100,
        "total_transactions": len(records),
        "date_range": {
            "from": min_date.isoformat(),
            "to": max_date.isoformat(),
        },
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "Brak pliku"}), 400
    file = request.files["file"]
    if not file.filename.lower().endswith(".csv"):
        return jsonify({"error": "Wymagany plik .csv"}), 400
    try:
        file_bytes = file.read()
        df = parse_revolut_csv(file_bytes)
        result = calculate_tax(df)
        return jsonify(result)
    except ValueError as e:
        return jsonify({"error": str(e)}), 422
    except Exception as e:
        return jsonify({"error": f"Błąd serwera: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
