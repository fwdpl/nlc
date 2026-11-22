"""2026-11-20 · v1.0 · generate_aurora_data

fotografwdrodze.pl — silnik ZORZY POLARNEJ (wpis /zorza-polarna/).
GitHub Actions co 15 min (repo fwdpl/nlc, katalog aurora/). Zapisuje:

  aurora.json          — stan „teraz” + prognoza na 3 noce dla 10 miast (co 15 min)
  aurora-history.json  — noce z burzami od 2015 r. + częstość wg lat (raz na dobę)

Źródła (każde z własnym rytmem odświeżania; przy awarii zostają dane z poprzedniego pliku):
  NOAA SWPC  planetary_k_index_1m.json          Kp szacowane co minutę          każdy przebieg
  NOAA SWPC  rtsw_mag_1m.json, rtsw_wind_1m.json Bz, Bt, prędkość, gęstość (L1) każdy przebieg
  NOAA SWPC  noaa-planetary-k-index-forecast    Kp 3-godz.: obserwowane + prognoza  każdy przebieg
  NOAA SWPC  27-day-outlook.txt                 największe Kp na 27 dni         co 6 h
  Open-Meteo zachmurzenie godzinowe, 10 miast   (jedno zapytanie)               co 1 h
  GFZ Potsdam Kp (definitywne + nowcast)        archiwum burz                   co 24 h

MODEL ZASIĘGU (orientacyjny — opisany wprost na stronie):
  szerokość geomagnetyczna miasta = szerokość dipolowa (biegun IGRF 2025: 80,8°N 72,7°W) − 3°
    (poprawka na szerokość skorygowaną CGM w Europie Środkowej; np. Belsk: dipol ~50,6°, CGM ~47,6°)
  granica owalu zorzy (nad głową) ≈ 66,5° − 2,05° × Kp   (klasyczna tabela NOAA/SWL: Kp5 → 56,3°, Kp9 → 48,1°)
  „nad głową”  : szer. geomag. ≥ granica
  „okiem”      : ≥ granica − 3°   (ok. 330 km na południe — zorza wysoko nad północnym horyzontem)
  „na zdjęciu” : ≥ granica − 6°   (ok. 650 km — czerwone górne partie nisko nad horyzontem, aparat)
  Kalibracja jakościowa: Kp5 = tylko wybrzeże na zdjęciu; Kp6 = północ i Wielkopolska na zdjęciu;
  Kp9 (10–11.05.2024) = cała Polska okiem, korona nad północną połową.

Użycie:
  python generate_aurora_data.py --prev aurora.json --prev-history aurora-history.json \
         --out aurora.new.json --out-history aurora-history.new.json
  --fixtures DIR   — dane z plików (testy offline), nazwy plików = klucze Source
  --now ISO        — udawany „teraz” (testy)
"""
import argparse
import datetime as dt
import json
import math
import os
import re
import sys
import time
import urllib.request
from zoneinfo import ZoneInfo

import numpy as np
from skyfield import almanac
from skyfield.api import Loader, load, wgs84
from skyfield_data import get_skyfield_data_path

VERSION = "1.0"
UA = "fotografwdrodze.pl aurora-bot (github.com/fwdpl/nlc)"

CITIES = {
    "bialystok": {"name": "Białystok", "lat": 53.1325, "lon": 23.1688},
    "gdansk":    {"name": "Gdańsk",    "lat": 54.3520, "lon": 18.6466},
    "krakow":    {"name": "Kraków",    "lat": 50.0647, "lon": 19.9450},
    "lodz":      {"name": "Łódź",      "lat": 51.7592, "lon": 19.4560},
    "lublin":    {"name": "Lublin",    "lat": 51.2465, "lon": 22.5684},
    "poznan":    {"name": "Poznań",    "lat": 52.4064, "lon": 16.9252},
    "rzeszow":   {"name": "Rzeszów",   "lat": 50.0413, "lon": 21.9990},
    "szczecin":  {"name": "Szczecin",  "lat": 53.4285, "lon": 14.5528},
    "warszawa":  {"name": "Warszawa",  "lat": 52.2297, "lon": 21.0122},
    "wroclaw":   {"name": "Wrocław",   "lat": 51.1079, "lon": 17.0385},
}
TZ_PL = ZoneInfo("Europe/Warsaw")
HISTORY_CITY = "poznan"          # „noc” w archiwum = ciemność w Poznaniu (środek kraju)

MODEL = {"pole_lat": 80.8, "pole_lon": -72.7, "cgm_offset": 3.0,
         "oval_a": 66.5, "oval_b": 2.05, "eye": 3.0, "camera": 6.0}
DARK_SUN = -12.0                 # „ciemno” = Słońce ≥ 12° pod horyzontem (koniec zmierzchu żeglarskiego)
NIGHTS = 3
CLOUD_OK, CLOUD_BAD = 40, 75     # % — najlepsze 3 h nocy: ≤40 sprzyja, >75 zasłania
HISTORY_FROM = 2015

U_KP1M = "https://services.swpc.noaa.gov/json/planetary_k_index_1m.json"
U_KPF = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index-forecast.json"
U_MAG = "https://services.swpc.noaa.gov/json/rtsw/rtsw_mag_1m.json"
U_WIND = "https://services.swpc.noaa.gov/json/rtsw/rtsw_wind_1m.json"
U_27D = "https://services.swpc.noaa.gov/text/27-day-outlook.txt"
U_METEO = ("https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
           "&hourly=cloud_cover&forecast_days=4&past_days=1&timeformat=unixtime&timezone=GMT")
U_GFZ = "https://kp.gfz.de/app/json/?start={s}T00:00:00Z&end={e}T23:59:59Z&index=Kp&status=all"

LOG = []


def log(msg):
    LOG.append(msg)
    print(msg, file=sys.stderr)


# --------------------------------------------------------------------- wejście --

class Source:
    def __init__(self, fixtures=None):
        self.fx = fixtures

    def get(self, key, url, text=False):
        if self.fx:
            p = os.path.join(self.fx, key)
            if not os.path.exists(p):
                raise FileNotFoundError(p)
            raw = open(p, encoding="utf-8").read()
        else:
            last = None
            for attempt in range(3):
                try:
                    req = urllib.request.Request(url, headers={"User-Agent": UA})
                    with urllib.request.urlopen(req, timeout=40) as r:
                        raw = r.read().decode("utf-8")
                    break
                except Exception as e:  # noqa: BLE001
                    last = e
                    time.sleep(3 * (attempt + 1))
            else:
                raise last
        return raw if text else json.loads(raw)


# ------------------------------------------------------------------- czas, model --

def iso(d):
    return d.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_t(s):
    s = s.strip().replace(" ", "T")
    if s.endswith("Z"):
        s = s[:-1]
    s = s.split(".")[0]
    return dt.datetime.fromisoformat(s).replace(tzinfo=dt.timezone.utc)


def mlat_of(lat, lon):
    """Szerokość geomagnetyczna (dipol) minus poprawka CGM dla Europy Środkowej."""
    pl, po = math.radians(MODEL["pole_lat"]), math.radians(MODEL["pole_lon"])
    la, lo = math.radians(lat), math.radians(lon)
    s = math.sin(la) * math.sin(pl) + math.cos(la) * math.cos(pl) * math.cos(lo - po)
    return math.degrees(math.asin(s)) - MODEL["cgm_offset"]


def oval(kp):
    return MODEL["oval_a"] - MODEL["oval_b"] * kp


def level(mlat, kp):
    """0 = nie, 1 = na zdjęciu (nisko na N), 2 = okiem, 3 = nad głową."""
    if kp is None:
        return 0
    d = mlat - oval(kp)
    if d >= 0:
        return 3
    if d >= -MODEL["eye"]:
        return 2
    if d >= -MODEL["camera"]:
        return 1
    return 0


def kp_needed(mlat, margin):
    """Najmniejsze Kp (zaokrąglone w górę do 1/3), przy którym miasto łapie dany poziom."""
    k = (MODEL["oval_a"] - margin - mlat) / MODEL["oval_b"]
    k = math.ceil(k * 3 - 1e-9) / 3
    return round(max(0.0, min(9.0, k)), 2)


def g_scale(kp):
    if kp is None:
        return 0
    for g, lim in ((5, 8.99), (4, 7.66), (3, 6.66), (2, 5.66), (1, 4.66)):
        if kp >= lim:
            return g
    return 0


# ------------------------------------------------------------------- efemeryda --

ts = load.timescale()
eph = Loader(get_skyfield_data_path())("de421.bsp")
SUN, MOON, EARTH = eph["sun"], eph["moon"], eph["earth"]


def sun_alt(lat, lon, times):
    obs = EARTH + wgs84.latlon(lat, lon)
    return obs.at(times).observe(SUN).apparent().altaz()[0].degrees


def moon_alt(lat, lon, times):
    obs = EARTH + wgs84.latlon(lat, lon)
    return obs.at(times).observe(MOON).apparent().altaz()[0].degrees


def dark_windows(lat, lon, t0, hours, step_min=5):
    """Lista (od, do) ciemności (Słońce ≤ DARK_SUN) w [t0, t0+hours]."""
    n = int(hours * 60 / step_min) + 1
    dts = [t0 + dt.timedelta(minutes=step_min * i) for i in range(n)]
    alt = sun_alt(lat, lon, ts.from_datetimes(dts))
    out, start = [], None
    for d, a in zip(dts, alt):
        if a <= DARK_SUN and start is None:
            start = d
        elif a > DARK_SUN and start is not None:
            out.append((start, d))
            start = None
    if start is not None:
        out.append((start, dts[-1]))
    return out


# ------------------------------------------------------------------- parsery --

def rows(j):
    """NOAA podaje produkty raz jako listę obiektów, raz jako tablicę z nagłówkiem — obsługa obu."""
    if isinstance(j, list) and j and isinstance(j[0], list):
        head = j[0]
        return [dict(zip(head, r)) for r in j[1:]]
    return j if isinstance(j, list) else []


def num(v):
    try:
        f = float(v)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


def parse_kp_forecast(j):
    out = []
    for r in rows(j):
        k = num(r.get("kp", r.get("Kp")))
        if k is None:
            continue
        st = str(r.get("observed", "")).lower()
        out.append({"t": iso(parse_t(r["time_tag"])), "kp": round(k, 2),
                    "s": "o" if st.startswith("obs") else "e" if st.startswith("est") else "p"})
    out.sort(key=lambda b: b["t"])
    return out


def parse_kp1m(j):
    rr = [r for r in rows(j) if num(r.get("estimated_kp")) is not None]
    if not rr:
        return None
    r = max(rr, key=lambda x: x["time_tag"])
    return {"kp": round(num(r["estimated_kp"]), 2), "t": iso(parse_t(r["time_tag"]))}


def active_series(j, fields):
    """RTSW: kilka satelitów na minutę — bierzemy oznaczony jako aktywny (active=true)."""
    out = {}
    for r in rows(j):
        if r.get("active") is not True:
            continue
        vals = {f: num(r.get(f)) for f in fields}
        if any(v is None for v in vals.values()):
            continue
        out[parse_t(r["time_tag"])] = vals
    return dict(sorted(out.items()))


def parse_27d(txt):
    m = re.search(r":Issued:\s*(\d{4} \w{3} \d{1,2} \d{4}) UTC", txt)
    issued = None
    if m:
        issued = iso(dt.datetime.strptime(m.group(1), "%Y %b %d %H%M").replace(tzinfo=dt.timezone.utc))
    days = []
    for line in txt.splitlines():
        mm = re.match(r"^\s*(\d{4})\s+(\w{3})\s+(\d{1,2})\s+(\d+)\s+(\d+)\s+(\d+)\s*$", line)
        if mm:
            d = dt.datetime.strptime(f"{mm.group(1)} {mm.group(2)} {mm.group(3)}", "%Y %b %d").date()
            days.append({"d": d.isoformat(), "kp": int(mm.group(6)), "ap": int(mm.group(5))})
    return {"issued": issued, "days": days} if days else None


def parse_meteo(j):
    """Open-Meteo dla wielu punktów zwraca listę; dla jednego — obiekt."""
    arr = j if isinstance(j, list) else [j]
    out = {}
    for key, loc in zip(CITIES.keys(), arr):
        h = loc.get("hourly", {})
        t, c = h.get("time", []), h.get("cloud_cover", [])
        if t and c:
            out[key] = {"t0": int(t[0]), "v": [None if v is None else int(round(v)) for v in c]}
    return out


# ------------------------------------------------------------------- liczenie --

def kp_in(blocks, a, b):
    """Największe Kp w blokach 3-godz. zachodzących na [a, b]; zwraca (kp, czas bloku, czy prognoza)."""
    best = None
    for bl in blocks:
        s = parse_t(bl["t"])
        e = s + dt.timedelta(hours=3)
        if e <= a or s >= b:
            continue
        if best is None or bl["kp"] > best[0]:
            best = (bl["kp"], bl["t"], bl["s"] == "p")
    return best


def cloud_at(cl, d):
    if not cl:
        return None
    i = int((d.timestamp() - cl["t0"]) // 3600)
    return cl["v"][i] if 0 <= i < len(cl["v"]) else None


def night_clouds(cl, a, b):
    """Średnie zachmurzenie nocy i najlepsze 3 godziny (start). None, gdy poza prognozą."""
    vals, hrs = [], []
    d = a.replace(minute=0, second=0, microsecond=0)
    while d < b:
        v = cloud_at(cl, d)
        if v is not None:
            vals.append(v)
            hrs.append(d)
        d += dt.timedelta(hours=1)
    if len(vals) < 3:
        return None
    best, bi = 999, 0
    for i in range(len(vals) - 2):
        m = sum(vals[i:i + 3]) / 3
        if m < best:
            best, bi = m, i
    return {"mean": int(round(sum(vals) / len(vals))), "best": int(round(best)), "best_from": iso(hrs[bi])}


def build_nights(blocks, clouds, now):
    out = {}
    t0 = now - dt.timedelta(hours=14)
    for key, c in CITIES.items():
        ml = c["mlat"]
        wins = [w for w in dark_windows(c["lat"], c["lon"], t0, 14 + 24 * (NIGHTS + 1)) if w[1] > now]
        nights = []
        for a, b in wins[:NIGHTS]:
            k = kp_in(blocks, a, b)
            kp = k[0] if k else None
            lv = level(ml, kp)
            ncl = night_clouds(clouds.get(key), max(a, now), b)
            # Księżyc: oświetlenie o północy nocy + ułamek ciemnych godzin nad horyzontem
            mid = a + (b - a) / 2
            tt = ts.from_datetimes([a + (b - a) * (i / 12) for i in range(13)])
            ma = moon_alt(c["lat"], c["lon"], tt)
            illum = float(almanac.fraction_illuminated(eph, "moon", ts.from_datetime(mid)))
            stars, verdict = lv, "none" if lv == 0 else "chance"
            if lv and ncl:
                if ncl["best"] > CLOUD_BAD:
                    stars, verdict = 0, "clouds"
                elif ncl["best"] > CLOUD_OK:
                    stars = max(1, lv - 1)
            nights.append({
                "date": (a.astimezone(TZ_PL) - dt.timedelta(hours=12)).date().isoformat(),
                "from": iso(a), "to": iso(b),
                "kp_max": kp, "kp_t": k[1] if k else None, "kp_forecast": bool(k and k[2]),
                "level": lv, "stars": stars, "verdict": verdict,
                "clouds": ncl,
                "moon": {"illum": round(illum, 2), "up": round(float((ma > 0).mean()), 2)},
            })
        out[key] = nights
    return out


def solar_wind(mag, wind, now):
    if not mag and not wind:
        return None
    res = {}
    if mag:
        tm = max(mag)
        res.update({"bz": round(mag[tm]["bz_gsm"], 1), "bt": round(mag[tm]["bt"], 1), "t": iso(tm)})
        rec = [v["bz_gsm"] for t, v in mag.items() if t > tm - dt.timedelta(minutes=30)]
        res["bz30"] = round(sum(rec) / len(rec), 1) if rec else None
    if wind:
        tw = max(wind)
        res.update({"v": int(round(wind[tw]["proton_speed"])), "n": round(wind[tw]["proton_density"], 1)})
        res.setdefault("t", iso(tw))
    # wykres 3 h, średnie 10-min
    spark = {"step": 10, "end": iso(now.replace(second=0, microsecond=0)), "bz": [], "v": []}
    for i in range(18, 0, -1):
        a, b = now - dt.timedelta(minutes=10 * i), now - dt.timedelta(minutes=10 * (i - 1))
        bz = [v["bz_gsm"] for t, v in (mag or {}).items() if a <= t < b]
        vv = [v["proton_speed"] for t, v in (wind or {}).items() if a <= t < b]
        spark["bz"].append(round(sum(bz) / len(bz), 1) if bz else None)
        spark["v"].append(int(round(sum(vv) / len(vv))) if vv else None)
    res["spark"] = spark
    return res


# ------------------------------------------------------------------- archiwum --

def build_history(src, prev, today):
    """Noce (wg ciemności w Poznaniu) z największym Kp ≥ 4,67. Przy istniejącym archiwum
    pobieramy tylko ostatnie 45 dni (statusy „now” zmieniają się w „def”)."""
    nights = dict(prev.get("nights", {})) if prev else {}
    start = dt.date(HISTORY_FROM, 1, 1) if not nights else today - dt.timedelta(days=45)
    times, kps, sts = [], [], []
    y = start
    while y <= today:
        e = min(dt.date(y.year, 12, 31), today)
        j = src.get(f"gfz_{y.isoformat()}.json", U_GFZ.format(s=y.isoformat(), e=e.isoformat()))
        times += j.get("datetime", [])
        kps += j.get("Kp", [])
        sts += j.get("status", ["def"] * len(j.get("datetime", [])))
        y = dt.date(y.year + 1, 1, 1)
    if not times:
        raise RuntimeError("GFZ: brak danych")
    c = CITIES[HISTORY_CITY]
    starts = [parse_t(t) for t in times]
    probe = []
    for s in starts:
        probe += [s, s + dt.timedelta(minutes=90), s + dt.timedelta(minutes=179)]
    alt = sun_alt(c["lat"], c["lon"], ts.from_datetimes(probe)).reshape(-1, 3)
    fresh = {}
    for s, k, st, a in zip(starts, kps, sts, alt):
        if k is None or min(a) > DARK_SUN:
            continue
        nd = (s.astimezone(TZ_PL) - dt.timedelta(hours=12)).date().isoformat()
        if nd < start.isoformat():
            continue
        if nd not in fresh or k > fresh[nd][0]:
            fresh[nd] = (round(float(k), 3), st)
    for nd in [n for n in nights if n >= start.isoformat()]:
        del nights[nd]
    for nd, (k, st) in fresh.items():
        if k >= 4.66:
            nights[nd] = {"kp": k, "def": st == "def"}
    nights = dict(sorted(nights.items()))
    years = {}
    for nd, v in nights.items():
        yy = years.setdefault(nd[:4], {"g1": 0, "g2": 0, "g3": 0, "g4": 0})
        g = g_scale(v["kp"])
        for lvl in range(1, min(g, 4) + 1):
            yy[f"g{lvl}"] += 1
    return {"version": VERSION, "generated_at": None, "source": "GFZ Potsdam, Kp (CC BY 4.0)",
            "night_city": HISTORY_CITY, "from": f"{HISTORY_FROM}-01-01",
            "nights": nights, "years": years}


# ------------------------------------------------------------------- główny --

def load_json(p):
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def age_h(d, key, now):
    t = (d or {}).get("updated", {}).get(key)
    return 1e9 if not t else (now - parse_t(t)).total_seconds() / 3600


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prev")
    ap.add_argument("--prev-history")
    ap.add_argument("--out", required=True)
    ap.add_argument("--out-history", required=True)
    ap.add_argument("--fixtures")
    ap.add_argument("--now")
    a = ap.parse_args()

    now = parse_t(a.now) if a.now else dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    src = Source(a.fixtures)
    prev = load_json(a.prev) if a.prev else None
    prev_h = load_json(a.prev_history) if a.prev_history else None
    upd = dict((prev or {}).get("updated", {}))

    for k, c in CITIES.items():
        ml = mlat_of(c["lat"], c["lon"])
        c["mlat"] = round(ml, 2)
        c["kp_camera"] = kp_needed(ml, MODEL["camera"])
        c["kp_eye"] = kp_needed(ml, MODEL["eye"])
        c["kp_overhead"] = kp_needed(ml, 0)

    # Kp 3-godz. (obserwowane + prognoza)
    blocks = None
    try:
        blocks = parse_kp_forecast(src.get("kp_forecast.json", U_KPF))
        upd["kp_forecast"] = iso(now)
    except Exception as e:  # noqa: BLE001
        log(f"Kp forecast: {e}")
    if not blocks:
        blocks = (prev or {}).get("kp_blocks_all", [])
    keep_from = now - dt.timedelta(hours=27)
    blocks_out = [b for b in blocks if parse_t(b["t"]) >= keep_from]

    # Kp teraz
    kp_now = None
    try:
        kp_now = parse_kp1m(src.get("kp_1m.json", U_KP1M))
    except Exception as e:  # noqa: BLE001
        log(f"Kp 1m: {e}")
    if not kp_now and prev:
        kp_now = (prev.get("now") or {}).get("kp")

    # wiatr słoneczny
    sw = None
    try:
        mag = active_series(src.get("rtsw_mag.json", U_MAG), ["bz_gsm", "bt"])
        wind = active_series(src.get("rtsw_wind.json", U_WIND), ["proton_speed", "proton_density"])
        sw = solar_wind(mag, wind, now)
    except Exception as e:  # noqa: BLE001
        log(f"RTSW: {e}")
    if not sw and prev:
        sw = (prev.get("now") or {}).get("sw")

    # chmury — co godzinę
    clouds = (prev or {}).get("clouds", {})
    if age_h(prev, "clouds", now) >= 0.95 or not clouds:
        try:
            lat = ",".join(str(c["lat"]) for c in CITIES.values())
            lon = ",".join(str(c["lon"]) for c in CITIES.values())
            cl = parse_meteo(src.get("meteo.json", U_METEO.format(lat=lat, lon=lon)))
            if len(cl) == len(CITIES):
                clouds = cl
                upd["clouds"] = iso(now)
        except Exception as e:  # noqa: BLE001
            log(f"Open-Meteo: {e}")

    # 27 dni — co 6 h
    outlook = (prev or {}).get("outlook27")
    if age_h(prev, "outlook27", now) >= 5.9 or not outlook:
        try:
            o = parse_27d(src.get("outlook27.txt", U_27D, text=True))
            if o:
                outlook = o
                upd["outlook27"] = iso(now)
        except Exception as e:  # noqa: BLE001
            log(f"27-day: {e}")

    # archiwum — raz na dobę
    hist = prev_h
    if not prev_h or (now - parse_t(prev_h.get("generated_at") or "2000-01-01T00:00:00Z")).total_seconds() > 23 * 3600:
        try:
            hist = build_history(src, prev_h, now.date())
            hist["generated_at"] = iso(now)
        except Exception as e:  # noqa: BLE001
            log(f"GFZ: {e}")

    past = [b for b in blocks_out if b["s"] != "p" and parse_t(b["t"]) >= now - dt.timedelta(hours=24)]
    fut = [b for b in blocks_out if parse_t(b["t"]) + dt.timedelta(hours=3) > now]
    nxt = max(fut, key=lambda b: b["kp"]) if fut else None
    data = {
        "version": VERSION,
        "generated_at": iso(now),
        "model": MODEL,
        "cities": CITIES,
        "now": {
            "kp": kp_now,
            "kp24_max": max((b["kp"] for b in past), default=None),
            "kp_next_max": {"kp": nxt["kp"], "t": nxt["t"]} if nxt else None,
            "sw": sw,
        },
        "kp_blocks": blocks_out,
        "kp_blocks_all": blocks[-90:],
        "nights": build_nights(blocks, clouds, now),
        "clouds": clouds,
        "outlook27": outlook,
        "last_storm": last_storm(hist),
        "updated": upd,
        "log": LOG,
    }
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
    if hist:
        with open(a.out_history, "w", encoding="utf-8") as f:
            json.dump(hist, f, ensure_ascii=False, separators=(",", ":"))
    print(f"OK {iso(now)} · Kp teraz {kp_now and kp_now['kp']} · log: {len(LOG)}")


def last_storm(h):
    """Ostatnia noc z Kp ≥ 5,67 (G2) i ≥ 6,67 (G3) — do licznika „kiedy ostatnio”."""
    if not h:
        return None
    out = {}
    for key, lim in (("g2", 5.66), ("g3", 6.66)):
        for nd in sorted(h["nights"], reverse=True):
            if h["nights"][nd]["kp"] >= lim:
                out[key] = {"date": nd, "kp": h["nights"][nd]["kp"]}
                break
    return out


if __name__ == "__main__":
    main()
