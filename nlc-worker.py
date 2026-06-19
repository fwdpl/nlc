#!/usr/bin/env python3
"""
fotografwdrodze.pl — worker NLC.

Co robi co noc:
  1. Znajduje najnowsze granule temperatury Aura/MLS (ML2T) przez CMR.
  2. Pobiera je z GES DISC (autoryzacja tokenem Earthdata).
  3. Wycina profile nad Polską, ekranuje jakość, uśrednia temperaturę
     na poziomie mezopauzy (~0.0046 hPa, ok. 83 km).
  4. Liczy frost point (z klimatologicznego H2O, bo subsystem 190-GHz MLS
     jest od 05.2024 zdegradowany), trend chłodzenia i czynnik słoneczny (F10.7).
  5. Wyznacza stan sezonu PO DACIE — dzięki temu każdy kolejny rok obsługuje się sam.
  6. Składa contract.json zgodny ze schematem widgetu.

Jeśli pobranie MLS się nie uda, worker odtwarza poprzedni kontrakt z większym
wiekiem danych i adnotacją „stale”, zamiast publikować pusty plik.

Wymagane zmienne środowiskowe:
  EARTHDATA_TOKEN  — bearer token z https://urs.earthdata.nasa.gov (Generate Token)
Opcjonalne:
  NLC_OUT          — ścieżka wyjściowa (domyślnie ./public/contract.json)
"""

import os, sys, json, math, tempfile, datetime as dt
from urllib.parse import urlencode

import requests

try:
    import numpy as np
    import h5py
except ImportError:
    print("Brak numpy/h5py — zainstaluj: pip install numpy h5py requests", file=sys.stderr)
    raise

# ----------------------------- KONFIGURACJA -----------------------------
CFG = {
    "short_name": "ML2T_NRT",      # NRT temperatury MLS: latencja ~3 h, 7 dni online (standard "ML2T" ma latencję tygodni!).
    "version": "",                 # puste = CMR dobierze najnowszą wersję NRT
    "poland_bbox": (49.0, 14.0, 55.0, 24.5),   # (lat_min, lon_min, lat_max, lon_max)
    "target_pressure_hpa": 0.0046, # poziom ~mezopauzy / strefy NLC (~83 km)
    "h2o_vmr": 4.0e-6,             # klimatologiczne H2O w mezopauzie [obj.], do frost pointu
    "trend_days": 3,               # ile dni wstecz (NRT trzyma 7 dni online)
    "max_granules": 350,           # górny limit pobieranych plików NRT na bieg (bez filtra przestrzennego CMR)
    # progi ekranowania jakości — NRT: pole Quality bywa zerowe (uproszczony model), więc je pomijamy
    "quality_min": -1e9,           # warunek Quality wyłączony (w NRT bywa 0 lub wartość wypełniająca)
    "convergence_max": 2.0,
    "temp_valid": (110.0, 260.0),
    # mapowania czynników
    "ftemp_k": 1.8,                # ostrość sigmoidy wokół frost pointu [K]
    "ftrend_scale": 4.0,           # K/dobę dające pełny rozjazd 0..1
    "f107_lo": 70.0, "f107_hi": 170.0,   # zakres F10.7 -> f_solar (1..0)
    # sezon (półkula N): start, peak_start, peak_end, end  (miesiąc, dzień)
    "season": ((5, 20), (6, 5), (7, 20), (8, 15)),
    "weights": {"w_T": 0.55, "w_d": 0.15, "w_lat": 0.20, "w_s": 0.10},
}
CMR = "https://cmr.earthdata.nasa.gov/search/granules.json"

# ----------------------------- FIZYKA -----------------------------
def ice_vapor_pressure_pa(T):
    """Ciśnienie pary nasyconej nad lodem (Marti & Mauersberger 1993), p w Pa, T w K."""
    return 10.0 ** (-2663.5 / T + 12.537)

def frost_point_k(p_h2o_pa):
    """Temperatura szronu dla danego ciśnienia cząstkowego pary wodnej."""
    return -2663.5 / (math.log10(p_h2o_pa) - 12.537)

def f_temp(T, Tfrost, k):
    return 1.0 / (1.0 + math.exp((T - Tfrost) / k))

def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))

def season_state(d):
    (sm, sd), (pm, pd), (em, ed), (xm, xd) = CFG["season"]
    y = d.year
    start = dt.date(y, sm, sd); pk0 = dt.date(y, pm, pd)
    pk1 = dt.date(y, em, ed);   end = dt.date(y, xm, xd)
    if d < start or d > end:
        return {"in_season": False, "phase": "off", "factor": 0.0}
    if d < pk0:
        f = (d - start).days / max(1, (pk0 - start).days); return {"in_season": True, "phase": "ramp", "factor": round(f, 3)}
    if d <= pk1:
        return {"in_season": True, "phase": "peak", "factor": 1.0}
    f = (end - d).days / max(1, (end - pk1).days)
    return {"in_season": True, "phase": "tail", "factor": round(f, 3)}

# ----------------------------- POBIERANIE MLS -----------------------------
def session():
    s = requests.Session()
    tok = os.environ.get("EARTHDATA_TOKEN", "").strip()
    if tok:
        s.headers["Authorization"] = "Bearer " + tok
    s.headers["User-Agent"] = "fwd-nlc-worker/2.0"
    return s

def find_granules(s, days):
    import time
    end = dt.datetime.now(dt.timezone.utc)
    start = end - dt.timedelta(days=days)
    q = {
        "short_name": CFG["short_name"],
        "temporal": f"{start:%Y-%m-%dT%H:%M:%SZ},{end:%Y-%m-%dT%H:%M:%SZ}",
        "page_size": 500, "sort_key": "-start_date",
        # UWAGA: bez bounding_box — CMR zwraca 500 dla granul NRT z filtrem przestrzennym.
        # Polskę odfiltrowujemy po pobraniu (temps_over_poland). Limit pobrań niżej w main.
    }
    if CFG["version"]:
        q["version"] = CFG["version"]
    url = CMR + "?" + urlencode(q)
    last = None
    for attempt in range(3):
        try:
            r = s.get(url, timeout=120); r.raise_for_status()
            out = []
            for e in r.json().get("feed", {}).get("entry", []):
                for ln in e.get("links", []):
                    h = ln.get("href", "")
                    if h.endswith(".he5") and "gesdisc" in h:
                        out.append({"date": e.get("time_start", "")[:10], "href": h})
                        break
            return out
        except Exception as ex:
            last = ex
            print(f"[worker] CMR próba {attempt + 1}/3 nieudana: {ex}", file=sys.stderr)
            time.sleep(5 * (attempt + 1))
    raise last

def read_he5(path):
    """Zwraca (lat[N], lon[N], pressure[L], T[N,L], status[N], quality[N], conv[N], prec[N,L])."""
    with h5py.File(path, "r") as f:
        g = f["/HDFEOS/SWATHS/Temperature"]
        T = g["Data Fields/Temperature"][:]
        status = g["Data Fields/Status"][:]
        quality = g["Data Fields/Quality"][:]
        conv = g["Data Fields/Convergence"][:]
        prec = g["Data Fields/L2gpPrecision"][:]
        lat = g["Geolocation Fields/Latitude"][:]
        lon = g["Geolocation Fields/Longitude"][:]
        pres = g["Geolocation Fields/Pressure"][:]
    return lat, lon, pres, T, status, quality, conv, prec

def temps_over_poland(path):
    lat, lon, pres, T, status, quality, conv, prec = read_he5(path)
    lvl = int(np.argmin(np.abs(pres - CFG["target_pressure_hpa"])))
    la0, lo0, la1, lo1 = CFG["poland_bbox"]
    tvmin, tvmax = CFG["temp_valid"]
    geo = (lat >= la0) & (lat <= la1) & (lon >= lo0) & (lon <= lo1)
    if geo.sum() == 0:
        return np.array([]), float(pres[lvl])
    # diagnostyka: ile profili nad Polską przechodzi każdy warunek z osobna
    g_status = geo & (status % 2 == 0)
    g_qual = g_status & (quality > CFG["quality_min"])
    g_conv = g_qual & (conv < CFG["convergence_max"])
    g_prec = g_conv & (prec[:, lvl] > 0)
    good = g_prec & (T[:, lvl] > tvmin) & (T[:, lvl] < tvmax)
    print(f"[worker]   nad PL={int(geo.sum())} status_ok={int(g_status.sum())} "
          f"+quality={int(g_qual.sum())} +conv={int(g_conv.sum())} +prec={int(g_prec.sum())} "
          f"final={int(good.sum())}", file=sys.stderr)
    return T[good, lvl], float(pres[lvl])

# ----------------------------- F10.7 (NOAA SWPC) -----------------------------
def fetch_f107(s):
    for url in ("https://services.swpc.noaa.gov/products/summary/10cm-flux.json",
                "https://services.swpc.noaa.gov/json/f107_cm_flux.json"):
        try:
            r = s.get(url, timeout=30); r.raise_for_status(); j = r.json()
            if isinstance(j, dict) and "Flux" in j:
                return float(j["Flux"])
            if isinstance(j, list) and j:
                return float(j[-1].get("flux") or j[-1].get("f107") or 0) or None
        except Exception:
            continue
    return None

# ----------------------------- SKŁADANIE KONTRAKTU -----------------------------
def build(temps_by_day, pressure_hpa, n_profiles, f107, now):
    days = sorted(temps_by_day)
    latest_day = days[-1]
    T = temps_by_day[latest_day]

    p_total_pa = pressure_hpa * 100.0
    Tfrost = frost_point_k(CFG["h2o_vmr"] * p_total_pa)
    f_t = f_temp(T, Tfrost, CFG["ftemp_k"])

    trend = None; f_tr = 0.5
    if len(days) >= 2:
        span = (dt.date.fromisoformat(latest_day) - dt.date.fromisoformat(days[0])).days or 1
        trend = (T - temps_by_day[days[0]]) / span
        f_tr = clamp(0.5 - trend / CFG["ftrend_scale"])

    f_s = None
    if f107 is not None:
        f_s = clamp((CFG["f107_hi"] - f107) / (CFG["f107_hi"] - CFG["f107_lo"]))

    obs_dt = dt.datetime.fromisoformat(latest_day + "T01:40:00")
    age_h = round((now - obs_dt).total_seconds() / 3600.0, 1)

    factors = {"f_temp": round(f_t, 3), "f_trend": round(f_tr, 3)}
    if f_s is not None:
        factors["f_solar"] = round(f_s, 3)

    return {
        "schema_version": "2.0",
        "generated_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "region": "PL",
        "source": {
            "temperature": f"Aura MLS {CFG['short_name']} v{CFG['version']}",
            "humidity": "MLS klimatologia (190-GHz zdegradowany od 05.2024)",
            "solar_activity": "NOAA SWPC F10.7",
        },
        "mesopause": {
            "temperature_k": round(T, 1),
            "frost_point_k": round(Tfrost, 1),
            "trend_k_per_day": (round(trend, 2) if trend is not None else None),
            "pressure_hpa": round(pressure_hpa, 5),
            "altitude_km": 83,
            "observed_at": obs_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "data_age_hours": age_h,
            "n_profiles": n_profiles,
        },
        "solar": {"f107": (round(f107, 1) if f107 is not None else None)},
        "season": season_state(now.date()),
        "factors": factors,
        "weights": CFG["weights"],
        "recent_sightings": [],
        "notes": "Wygenerowane automatycznie. f_lat liczony po stronie klienta.",
    }

def degrade_previous(prev, now):
    """Gdy brak świeżych danych MLS — odtwórz poprzedni kontrakt ze świeżym sezonem i wiekiem."""
    prev = dict(prev)
    prev["generated_at"] = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    prev["season"] = season_state(now.date())   # sezon zawsze aktualny, niezależnie od MLS
    m = prev.get("mesopause") or {}
    if m.get("observed_at"):
        try:
            obs = dt.datetime.strptime(m["observed_at"], "%Y-%m-%dT%H:%M:%SZ")
            m["data_age_hours"] = round((now - obs).total_seconds() / 3600.0, 1)
        except Exception:
            pass
    prev["mesopause"] = m
    prev["notes"] = "STALE: brak świeżych danych MLS — odtworzono poprzedni odczyt, sezon zaktualizowany."
    return prev

# ----------------------------- MAIN -----------------------------
def main():
    out = os.environ.get("NLC_OUT", "public/contract.json")
    now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    s = session()

    try:
        grans = find_granules(s, CFG["trend_days"])
        if not grans:
            raise RuntimeError(f"CMR nie zwrócił granul {CFG['short_name']} w oknie czasowym.")
        print(f"[worker] CMR: {len(grans)} granul; pobieram do {CFG['max_granules']}, filtruję Polskę po pobraniu.")
        temps_lists, last_pres, hits = {}, CFG["target_pressure_hpa"], 0
        with tempfile.TemporaryDirectory() as td:
            for g in grans[:CFG["max_granules"]]:
                fp = os.path.join(td, os.path.basename(g["href"]))
                try:
                    r = s.get(g["href"], timeout=180, allow_redirects=True)
                    r.raise_for_status()
                    with open(fp, "wb") as fh:
                        fh.write(r.content)
                    vals, last_pres = temps_over_poland(fp)
                    if vals.size:
                        temps_lists.setdefault(g["date"], []).extend(vals.tolist())
                        hits += 1
                except Exception as ex:
                    print(f"[worker] pominięto granulę ({ex})", file=sys.stderr)
                finally:
                    if os.path.exists(fp):
                        os.remove(fp)
        print(f"[worker] przeloty z profilami nad Polską: {hits}")
        temps = {d: sum(v) / len(v) for d, v in temps_lists.items() if v}
        last_n = sum(len(v) for v in temps_lists.values())
        if not temps:
            raise RuntimeError("Brak profili MLS przechodzących ekranowanie nad Polską.")
        f107 = fetch_f107(s)
        contract = build(temps, last_pres, last_n, f107, now)
    except Exception as ex:
        print(f"[worker] Pobranie MLS nieudane: {ex}", file=sys.stderr)
        if os.path.exists(out):
            with open(out, encoding="utf-8") as fh:
                contract = degrade_previous(json.load(fh), now)
        else:
            # pierwszy bieg bez danych: sam sezon + okno liczy klient
            contract = {
                "schema_version": "2.0",
                "generated_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "region": "PL", "season": season_state(now.date()),
                "factors": {}, "weights": CFG["weights"],
                "mesopause": {}, "solar": {}, "recent_sightings": [],
                "notes": "Brak danych MLS przy pierwszym uruchomieniu.",
            }

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(contract, fh, ensure_ascii=False, indent=2)
    print(f"[worker] Zapisano {out}: T={contract.get('mesopause', {}).get('temperature_k')} "
          f"sezon={contract['season']['phase']}")

if __name__ == "__main__":
    main()
