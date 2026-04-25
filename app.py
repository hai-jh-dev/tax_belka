import os
import io
import json
from datetime import date, timedelta

import pandas as pd
import requests
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB for bulk uploads

TAX_RATE = 0.19
NBP_RANGE_API = "https://api.nbp.pl/api/exchangerates/rates/a/eur/{start}/{end}/?format=json"
NBP_SINGLE_API = "https://api.nbp.pl/api/exchangerates/rates/a/eur/{date}/?format=json"

# Persistent disk cache: nbp_cache.json lives next to app.py
CACHE_FILE = os.path.join(os.path.dirname(__file__), "nbp_cache.json")


def _load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_cache(cache):
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f, separators=(",", ":"))
    except Exception as e:
        print(f"[warn] Cannot save NBP cache: {e}")


# In-memory layer (loaded once at startup, written to disk after each NBP fetch)
_rate_cache = _load_cache()
print(f"[nbp cache] Loaded {len(_rate_cache)} cached rates from disk.")


def _get_cached(date_str):
    return _rate_cache.get(date_str)


def _set_cached(date_str, rate):
    _rate_cache[date_str] = rate
    _save_cache(_rate_cache)


def fetch_single_rate(date_str):
    cached = _get_cached(date_str)
    if cached is not None:
        return cached
    try:
        r = requests.get(NBP_SINGLE_API.format(date=date_str), timeout=8)
        if r.status_code == 200:
            rate = r.json()["rates"][0]["mid"]
            _set_cached(date_str, rate)
            return rate
    except Exception:
        pass
    return None


def get_nbp_rates_bulk(start, end):
    """Fetch a date range from NBP, filling in any gaps from cache. Returns dict {date_str: rate}."""
    # Figure out which dates we're missing from cache
    result = {}
    missing_start = None
    missing_end = None

    check = start
    while check <= end:
        ds = check.isoformat()
        cached = _get_cached(ds)
        if cached is not None:
            result[ds] = cached
        else:
            if missing_start is None:
                missing_start = check
            missing_end = check
        check += timedelta(days=1)

    # Fetch only the missing range from NBP (if any)
    if missing_start is not None:
        url = NBP_RANGE_API.format(start=missing_start.isoformat(), end=missing_end.isoformat())
        try:
            r = requests.get(url, timeout=15)
            if r.status_code == 200:
                fetched = {}
                for row in r.json()["rates"]:
                    fetched[row["effectiveDate"]] = row["mid"]
                    result[row["effectiveDate"]] = row["mid"]
                # Persist new rates
                _rate_cache.update(fetched)
                _save_cache(_rate_cache)
                print(f"[nbp cache] Fetched {len(fetched)} new rates ({missing_start} – {missing_end}), cache now {len(_rate_cache)} entries.")
        except Exception as e:
            print(f"[warn] NBP bulk fetch failed: {e}")

    return result


def prev_business_day_rate(tx_date, rate_map):
    """Return (rate, rate_date_str) from the previous business day."""
    for delta in range(1, 12):
        d = tx_date - timedelta(days=delta)
        rate = rate_map.get(d.isoformat())
        if rate is not None:
            return rate, d.isoformat()
    # Fallback: single requests (also cached)
    for delta in range(1, 12):
        d = tx_date - timedelta(days=delta)
        rate = fetch_single_rate(d.isoformat())
        if rate is not None:
            return rate, d.isoformat()
    return None, None


def parse_money(val):
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
    except Exception:
        pass
    s = str(val).replace("+", "").replace("€", "").replace(",", "").strip()
    try:
        return float(s)
    except ValueError:
        return None


def parse_revolut_csv(file_bytes, filename="unknown"):
    df = None
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            df = pd.read_csv(io.BytesIO(file_bytes), encoding=enc)
            break
        except Exception:
            continue
    if df is None:
        raise ValueError(f"[{filename}] Nie można odczytać pliku CSV.")

    df.columns = [c.strip() for c in df.columns]

    required = {"Completed Date", "Description", "Money in"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"[{filename}] Nieoczekiwany format. Brakuje kolumn: {missing}. "
            f"Dostępne: {list(df.columns)}"
        )

    interest_df = df[df["Description"].str.contains("Gross interest", na=False, case=False)].copy()

    if interest_df.empty:
        raise ValueError(f"[{filename}] Brak wierszy 'Gross interest'.")

    interest_df["date"] = pd.to_datetime(
        interest_df["Completed Date"].str.strip(), format="%d %b %Y", errors="coerce"
    ).dt.date

    interest_df["amount_eur"] = interest_df["Money in"].apply(parse_money)

    bad = interest_df[interest_df["date"].isna() | interest_df["amount_eur"].isna()]
    if not bad.empty:
        print(f"[warn][{filename}] Pominięto {len(bad)} wierszy z błędem parsowania")

    interest_df = interest_df.dropna(subset=["date", "amount_eur"])
    interest_df = interest_df[interest_df["amount_eur"] > 0]

    if interest_df.empty:
        raise ValueError(f"[{filename}] Po filtrowaniu brak prawidłowych wierszy.")

    interest_df = interest_df[["date", "amount_eur"]].reset_index(drop=True)
    interest_df["source"] = filename
    return interest_df


def calculate_tax(interest_df):
    min_date = interest_df["date"].min()
    max_date = interest_df["date"].max()

    rate_map = get_nbp_rates_bulk(min_date - timedelta(days=15), max_date)

    records = []
    missing_rates = []

    for _, row in interest_df.iterrows():
        tx_date = row["date"]
        amount_eur = float(row["amount_eur"])
        source = row.get("source", "")

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
            "source": source,
        })

    if not records:
        raise ValueError("Nie udało się pobrać kursów NBP. Sprawdź połączenie z internetem.")

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

    # Per-file summary
    per_file = (
        result_df.groupby("source")
        .agg(
            transactions=("amount_eur", "count"),
            amount_eur=("amount_eur", "sum"),
            amount_pln=("amount_pln", "sum"),
            tax_pln=("tax_pln", "sum"),
        )
        .reset_index()
    )

    yearly = {
        "amount_eur": round(float(result_df["amount_eur"].sum()), 4),
        "amount_pln": round(float(result_df["amount_pln"].sum()), 4),
        "tax_pln": round(float(result_df["tax_pln"].sum()), 4),
    }

    return {
        "daily": daily.round(4).to_dict(orient="records"),
        "monthly": monthly.round(4).to_dict(orient="records"),
        "per_file": per_file.round(4).to_dict(orient="records"),
        "yearly": yearly,
        "missing_rates": missing_rates,
        "tax_rate_pct": TAX_RATE * 100,
        "total_transactions": len(records),
        "date_range": {
            "from": min_date.isoformat(),
            "to": max_date.isoformat(),
        },
        "cache_size": len(_rate_cache),
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    files = request.files.getlist("files")
    if not files or all(f.filename == "" for f in files):
        return jsonify({"error": "Brak plików"}), 400

    all_frames = []
    file_errors = []

    for f in files:
        if not f.filename.lower().endswith(".csv"):
            file_errors.append(f"{f.filename}: pomijam (nie .csv)")
            continue
        try:
            df = parse_revolut_csv(f.read(), f.filename)
            all_frames.append(df)
        except ValueError as e:
            file_errors.append(str(e))

    if not all_frames:
        return jsonify({"error": "Żaden plik nie zawierał danych. " + " | ".join(file_errors)}), 422

    combined = pd.concat(all_frames, ignore_index=True)
    # Deduplicate: same date + same amount from different files (overlap guard)
    combined = combined.drop_duplicates(subset=["date", "amount_eur", "source"])

    try:
        result = calculate_tax(combined)
        result["file_errors"] = file_errors
        result["files_loaded"] = [df["source"].iloc[0] for df in all_frames]
        return jsonify(result)
    except ValueError as e:
        return jsonify({"error": str(e)}), 422
    except Exception as e:
        return jsonify({"error": f"Błąd serwera: {str(e)}"}), 500


@app.route("/cache-info")
def cache_info():
    return jsonify({"cached_rates": len(_rate_cache), "sample": dict(list(_rate_cache.items())[:5])})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
