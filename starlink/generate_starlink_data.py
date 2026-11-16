"""2026-11-16 · v1.1 · generate_starlink_data.py
v1.1: pole net_precision przy startach (exact/day/week/month/coarse) z Launch Library 2 — dla startów
      bez ustalonego dnia LL2 podaje NET = koniec miesiąca; karta pokazuje wtedy sam miesiąc.
      Starty z precyzją miesiąca/tygodnia trzymamy na liście do 45 dni (NET to górna granica).
      User-Agent bez adresu e-mail (adres strony zamiast kontaktu).
Generator danych dla wpisu /jak-odnalezc-satelity-starlink-nad-polska/ (silnik fws-).

Co robi (uruchamiany co 3 h, np. w GitHub Actions):
  1. Launch Library 2 — nadchodzące i ostatnie starty Starlink (2 zapytania; limit darmowy 15/h).
  2. CelesTrak GP (last-30-days) — elementy orbit obiektów STARLINK z ostatnich startów.
  3. CelesTrak SATCAT — data startu każdej grupy (INTDES), żeby dopasować grupę do startu z LL2.
  4. CelesTrak SupGP FILE=starlink-gXX-YY — dane od SpaceX (obiekt „STACK” tuż po starcie,
     ewentualnie przed startem). Nazewnictwo potwierdzone 06.10.2026 na starlink-g15-27.
  5. Skyfield/SGP4 — widoczne przeloty POCIĄGÓW nad 10 miastami, 7 dni do przodu + 1 dzień wstecz.
  6. Zapis starlink.json.

Definicje (opisane też w treści wpisu):
  * Przelot widoczny = satelita ≥ 10° nad horyzontem, oświetlony przez Słońce,
    a u obserwatora Słońce ≤ −6° (koniec zmierzchu cywilnego).
  * Pociąg = co najmniej MIN_TRAIN_SATS satelitów z jednego startu, które przechodzą
    jeden za drugim (przerwa między kolejnymi ≤ TRAIN_GAP_S). Grupy, które się już
    rozjechały po orbicie, pociągu nie tworzą i nie trafiają do tabeli.
  * Tylko grupy ≤ MAX_AGE_DAYS od startu.
  * Ocena ★ (1–3) to kategoria, nie jasność: wysokość przelotu, wiek pociągu, jasność nieba.
    Jasności w magnitudo nie podajemy — nie ma wiarygodnego modelu dla świeżych satelitów.

Weryfikacja (06.10.2026): przeloty pojedynczych satelitów grup 2026-219 (G15-27)
porównane z Heavens-Above — Poznań (NORAD 100753, 5 przelotów) i Kraków (100766, 5 przelotów):
początek ±1–3 s, maks. wysokość ±1°, kierunki zgodne. Bramka (±1 min, ±5°, ±3°) zaliczona.

Użycie:
  python generate_starlink_data.py                     # dane z sieci
  python generate_starlink_data.py --fixtures DIR      # dane z plików (test offline)
  python generate_starlink_data.py --now 2026-10-11T12:00Z   # udawany „teraz” (test)
"""

import argparse
import json
import math
import os
import re
import sys
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
from skyfield.api import load, wgs84, EarthSatellite, Loader

VERSION = "1.1"
OUT_PATH = "starlink.json"
TZ_PL = ZoneInfo("Europe/Warsaw")
UA = "fotografwdrodze.pl starlink-widget (+https://fotografwdrodze.pl/)"

MAX_AGE_DAYS = 10          # pociąg starszy niż tyle dni = już się rozjechał
PREDICT_DAYS = 7           # ile dni do przodu
PAST_DAYS = 1              # ile dni wstecz (pod „co to było?” w v2)
MIN_ALT = 10.0             # próg horyzontu dla przelotu
LIST_MIN_MAX_ALT = 20.0    # przelotów niższych niż tyle nie pokazujemy
SUN_MAX = -6.0             # Słońce u obserwatora najwyżej tyle
TRAIN_GAP_S = 240          # max przerwa między kolejnymi satelitami w pociągu
MIN_TRAIN_SATS = 5
MAX_SATS_PER_GROUP = 12    # próbka satelitów z grupy (zawsze z pierwszym i ostatnim) — szybkość
LAUNCH_HORIZON_DAYS = 14   # starty w liście „najbliższe starty”
LAUNCH_HORIZON_COARSE_DAYS = 45   # gdy znany jest tylko tydzień/miesiąc
SHELL_HEIGHT_KM = 350      # wysokość do zdania o oknach oświetlenia (świeże pociągi ~280–360 km)

# Nachylenie wg numeru grupy — HEURYSTYKA na czas przed startem, gdy nie ma danych orbity.
# Potwierdzone na danych CelesTrak 06.10.2026: grupa 15 = 70,0°; Starship 31-1 (2026-225) = 30,5°.
# Pozostałe z wiedzy ogólnej — generator zawsze woli nachylenie z danych.
GROUP_INC = {6: 43.0, 12: 43.0, 10: 53.0, 11: 53.0, 15: 70.0, 17: 97.6, 31: 30.5}

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

LL2 = "https://ll.thespacedevs.com/2.3.0/launches/{}/?search=starlink&limit=12&mode=normal"
GP_LAST30 = "https://celestrak.org/NORAD/elements/gp.php?GROUP=last-30-days&FORMAT=json"
GP_STARLINK = "https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=json"
SATCAT = "https://celestrak.org/satcat/records.php?INTDES={}&FORMAT=json"
SUPGP = "https://celestrak.org/NORAD/elements/supplemental/sup-gp.php?FILE={}&FORMAT=json"

# --------------------------------------------------------------------- wejście --

class Source:
    """Pobiera z sieci albo z katalogu fixtures (nazwy plików = klucze)."""

    def __init__(self, fixtures=None):
        self.fx = fixtures
        self.log = []

    def get(self, key, url):
        if self.fx:
            p = os.path.join(self.fx, key + ".json")
            if not os.path.exists(p):
                self.log.append(f"brak fixture {key}")
                return None
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as r:
                txt = r.read().decode("utf-8")
            return json.loads(txt)
        except Exception as e:   # CelesTrak przy braku pliku zwraca tekst, nie JSON
            self.log.append(f"{key}: {type(e).__name__}")
            return None


# ------------------------------------------------------------------ pomocnicze --

ts = load.timescale()
try:
    from skyfield_data import get_skyfield_data_path
    eph = Loader(get_skyfield_data_path())("de421.bsp")
except ImportError:
    eph = load("de421.bsp")
sun, earth = eph["sun"], eph["earth"]

DIRS_PL = ["północy", "północnego wschodu", "wschodu", "południowego wschodu",
           "południa", "południowego zachodu", "zachodu", "północnego zachodu"]
DIRS_TO = ["północ", "północny wschód", "wschód", "południowy wschód",
           "południe", "południowy zachód", "zachód", "północny zachód"]
DIRS_ABBR = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


def d8(az):
    return int(((az + 22.5) % 360) // 45)


def iso(t):
    """Czas skyfield → ISO UTC z dokładnością do sekundy."""
    return t.utc_datetime().replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_iso(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def group_from_name(name):
    """'Falcon 9 Block 5 | Starlink Group 15-25' → (15, 25, 'starlink-g15-25')."""
    m = re.search(r"Group\s+(\d+)-(\d+)", name or "")
    if not m:
        return None, None, None
    g, n = int(m.group(1)), int(m.group(2))
    return g, n, f"starlink-g{g}-{n}"


def net_precision(L):
    """Precyzja daty startu z LL2 → exact / day / week / month / coarse."""
    name = ((L.get("net_precision") or {}).get("name") or "").lower()
    if not name:
        st = (L.get("status") or {}).get("abbrev")
        return "day" if st in ("TBD", "TBC") else "exact"
    if any(w in name for w in ("second", "minute", "hour")):
        return "exact"
    if any(w in name for w in ("day", "morning", "afternoon", "evening", "night")):
        return "day"
    if "week" in name:
        return "week"
    if "month" in name:
        return "month"
    return "coarse"


def visibility_verdict(inc):
    if inc is None:
        return "unknown", "nie wiadomo — sprawdzimy po starcie"
    if inc >= 50:
        return "yes", f"tak — orbita ok. {inc:.0f}°, przeloty wysoko nad Polską"
    if inc >= 38:
        return "low", f"słabo — orbita ok. {inc:.0f}°, pociąg najwyżej ok. 10–15° nad południowym horyzontem"
    return "no", f"nie — orbita ok. {inc:.0f}°, pociąg nie wzniesie się nad horyzont Polski"


def sat_from_omm(o):
    f = {k: str(o[k]) for k in ("OBJECT_NAME", "OBJECT_ID", "EPOCH", "MEAN_MOTION", "ECCENTRICITY",
                                "INCLINATION", "RA_OF_ASC_NODE", "ARG_OF_PERICENTER", "MEAN_ANOMALY",
                                "NORAD_CAT_ID", "BSTAR", "MEAN_MOTION_DOT", "MEAN_MOTION_DDOT")}
    f.update(EPHEMERIS_TYPE="0", CLASSIFICATION_TYPE="U", ELEMENT_SET_NO="999", REV_AT_EPOCH="0")
    return EarthSatellite.from_omm(ts, f)


def height_km(mean_motion):
    mu = 398600.4418
    n = mean_motion * 2 * math.pi / 86400
    return (mu / n ** 2) ** (1 / 3) - 6378.137


# ---------------------------------------------------------------- przeloty --

def visible_passes(sat, topos, t0, t1):
    """Widoczne odcinki przelotów jednego satelity: lista dict z czasami i kierunkami."""
    t, ev = sat.find_events(topos, t0, t1, altitude_degrees=MIN_ALT)
    out, i = [], 0
    ev = list(ev)
    while i < len(ev):
        if ev[i] != 0:
            i += 1
            continue
        try:
            j = ev.index(2, i)
        except ValueError:
            break
        n = max(20, int((t[j] - t[i]) * 86400 / 5))         # próbka co ~5 s
        tt = ts.linspace(t[i], t[j], n)
        alt, az, _ = (sat - topos).at(tt).altaz()
        lit = sat.at(tt).is_sunlit(eph)
        sun_alt = (earth + topos).at(tt).observe(sun).apparent().altaz()[0].degrees
        vis = lit & (sun_alt <= SUN_MAX)
        if vis.any():
            k = np.where(vis)[0]
            a, b = k[0], k[-1]
            m = k[np.argmax(alt.degrees[k])]
            out.append({
                "t_start": tt[a], "t_end": tt[b], "t_max": tt[m],
                "az_start": float(az.degrees[a]), "az_end": float(az.degrees[b]),
                "alt_start": float(alt.degrees[a]), "alt_end": float(alt.degrees[b]),
                "alt_max": float(alt.degrees[m]), "az_max": float(az.degrees[m]),
                "sun_alt": float(sun_alt[a]),
                "fade": bool(b < len(tt) - 1 and not lit[b + 1]),     # znika w cieniu Ziemi
                "emerge": bool(a > 0 and not lit[a - 1]),             # wyłania się z cienia
            })
        i = j + 1
    return out


def ground_track(sat, t_a, t_b):
    """Punkty podsatelitarne co ~15 s — do mapy przelotu."""
    n = max(8, int((t_b - t_a) * 86400 / 15))
    tt = ts.linspace(t_a, t_b, n)
    sp = wgs84.subpoint_of(sat.at(tt))
    return [[round(float(la), 2), round(float(lo), 2)] for la, lo in zip(sp.latitude.degrees, sp.longitude.degrees)]


def stars(alt_max, age_days, sun_alt):
    s = 3 if alt_max >= 60 else 2 if alt_max >= 35 else 1
    if age_days is not None and age_days > 5:
        s -= 1            # pociąg się rozciąga i podnosi orbitę — mniej efektowny
    if sun_alt > -9:
        s -= 1            # jasne niebo zaraz po zmierzchu / przed świtem
    return max(1, s)


def train_events(group, topos, t0, t1, now_dt):
    """Łączy przeloty satelitów jednej grupy w pociągi."""
    sats = group["sats"]
    n_all = len(sats)
    if len(sats) > MAX_SATS_PER_GROUP:
        idx = np.linspace(0, len(sats) - 1, MAX_SATS_PER_GROUP).astype(int)
        sats = [sats[i] for i in idx]
    single = group.get("single_stack", False)
    passes = []
    for s in sats:
        for p in visible_passes(s, topos, t0, t1):
            p["sat"] = s
            passes.append(p)
    passes.sort(key=lambda p: p["t_start"].tt)
    clusters, cur = [], []
    for p in passes:
        if cur and (p["t_start"].tt - cur[-1]["t_start"].tt) * 86400 > TRAIN_GAP_S:
            clusters.append(cur)
            cur = []
        cur.append(p)
    if cur:
        clusters.append(cur)
    need = 1 if single else min(MIN_TRAIN_SATS, max(2, len(sats) // 3))
    events = []
    for c in clusters:
        if len(c) < need:
            continue
        lead, last = c[0], max(c, key=lambda p: p["t_end"].tt)
        top = max(c, key=lambda p: p["alt_max"])
        if top["alt_max"] < LIST_MIN_MAX_ALT:
            continue
        launch = group.get("launch_dt")
        age = (lead["t_start"].utc_datetime() - launch).total_seconds() / 86400 if launch else None
        events.append({
            "group": group["key"],
            "start": iso(lead["t_start"]),
            "end": iso(last["t_end"]),
            "max_time": iso(top["t_max"]),
            "from_az": round(lead["az_start"]), "to_az": round(lead["az_end"]),
            "from": DIRS_ABBR[d8(lead["az_start"])], "to": DIRS_ABBR[d8(lead["az_end"])],
            "from_pl": DIRS_PL[d8(lead["az_start"])], "to_pl": DIRS_TO[d8(lead["az_end"])],
            "max_alt": round(top["alt_max"]),
            "max_az": round(top["az_max"]),
            "start_alt": round(lead["alt_start"]),
            "fades": lead["fade"],
            "emerges": lead["emerge"],
            "sun_alt": round(lead["sun_alt"], 1),
            "n_sats": (min(n_all, round(len(c) / len(sats) * n_all)) if not single else None),
            "age_days": round(age, 1) if age is not None else None,
            "stars": stars(top["alt_max"], age, lead["sun_alt"]),
            "forecast": group.get("forecast", False),
            "track": ground_track(lead["sat"], lead["t_start"], lead["t_end"]),
        })
    return events


def light_windows(topos, day_local):
    """Okna, w których satelity na SHELL_HEIGHT_KM świecą nad głową, a niebo jest ciemne:
    Słońce między −6° a −dep (dep = kąt, przy którym cień Ziemi sięga tej wysokości).
    Zwraca wieczór i poranek następnego dnia albo all_night=True."""
    dep = math.degrees(math.acos(6371.0 / (6371.0 + SHELL_HEIGHT_KM)))
    start = datetime(day_local.year, day_local.month, day_local.day, 12, tzinfo=TZ_PL)
    tt = ts.from_datetimes([start + timedelta(minutes=5 * k) for k in range(24 * 12 + 1)])
    sa = (earth + topos).at(tt).observe(sun).apparent().altaz()[0].degrees
    ok = (sa <= SUN_MAX) & (sa >= -dep)
    dark = sa <= SUN_MAX
    def spans(mask):
        res, run = [], None
        for k, v in enumerate(mask):
            if v and run is None:
                run = k
            if not v and run is not None:
                res.append((run, k - 1)); run = None
        if run is not None:
            res.append((run, len(mask) - 1))
        return res
    fmt = lambda k: (start + timedelta(minutes=5 * k)).astimezone(TZ_PL).strftime("%H:%M")
    all_night = bool(dark.any() and not (dark & ~ok).any())
    w = [{"from": fmt(a), "to": fmt(b)} for a, b in spans(ok)]
    return {"date": day_local.isoformat(), "all_night": all_night, "windows": w,
            "shell_km": SHELL_HEIGHT_KM, "sun_dep": round(dep, 1)}


# ------------------------------------------------------------------- główne --

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures")
    ap.add_argument("--now")
    ap.add_argument("--out", default=OUT_PATH)
    ap.add_argument("--prev", default=None, help="poprzedni starlink.json (liczba satelitów raz na dobę)")
    a = ap.parse_args()
    src = Source(a.fixtures)
    now = parse_iso(a.now) if a.now else datetime.now(timezone.utc).replace(microsecond=0)

    # 1. starty
    up = src.get("ll_upcoming", LL2.format("upcoming")) or {}
    prev = src.get("ll_previous", LL2.format("previous")) or {}

    def launch_row(L):
        g, n, f = group_from_name(L.get("name"))
        pad = (L.get("pad") or {})
        return {"name": (L.get("mission") or {}).get("name") or L.get("name"),
                "net": L.get("net"), "status": (L.get("status") or {}).get("abbrev"),
                "status_name": (L.get("status") or {}).get("name"),
                "site": ((pad.get("location") or {}).get("name") or "").split(",")[0],
                "net_precision": net_precision(L),
                "group": g, "num": n, "file": f, "inc": None, "inc_source": None}

    upcoming = [launch_row(L) for L in up.get("results", [])
                if "starlink" in (L.get("name") or "").lower()]
    upcoming = [r for r in upcoming if r["net"] and
                parse_iso(r["net"]) <= now + timedelta(days=(LAUNCH_HORIZON_COARSE_DAYS
                    if r["net_precision"] in ("week", "month", "coarse") else LAUNCH_HORIZON_DAYS)) and
                parse_iso(r["net"]) >= now - timedelta(hours=6)]
    recent = [launch_row(L) for L in prev.get("results", [])
              if "starlink" in (L.get("name") or "").lower()
              and (L.get("status") or {}).get("abbrev") == "Success"]
    recent = [r for r in recent if r["net"] and now - parse_iso(r["net"]) <= timedelta(days=MAX_AGE_DAYS)]

    # 2. orbity grup z ostatnich startów
    gp = src.get("gp_last30", GP_LAST30) or []
    by_intdes = {}
    for o in gp:
        if "STARLINK" not in o.get("OBJECT_NAME", "") and "TBA" not in o.get("OBJECT_NAME", ""):
            continue
        by_intdes.setdefault(o["OBJECT_ID"][:8], []).append(o)

    groups = []
    for intdes, objs in by_intdes.items():
        sc = src.get("satcat_" + intdes, SATCAT.format(intdes)) or []
        ldate = sc[0].get("LAUNCH_DATE") if sc else None
        # dopasowanie do startu LL2 po dacie (UTC)
        match = None
        for r in recent:
            if ldate and r["net"][:10] == ldate:
                match = r
        if not match:
            continue                  # nie Starlink z ostatnich dni (albo za stary)
        star = [o for o in objs if "STARLINK" in o["OBJECT_NAME"]] or objs
        inc = float(np.median([o["INCLINATION"] for o in star]))
        h = float(np.median([height_km(o["MEAN_MOTION"]) for o in star]))
        match["inc"], match["inc_source"], match["intdes"] = round(inc, 1), "orbit", intdes
        groups.append({"key": match["name"], "intdes": intdes, "launch_dt": parse_iso(match["net"]),
                       "inc": inc, "height_km": round(h), "sats": [sat_from_omm(o) for o in star],
                       "forecast": False})

    # 3. dane SpaceX (SupGP) dla świeżych startów bez GP i dla startów ≤ 72 h
    for r in recent + upcoming:
        if not r["file"] or any(g["key"] == r["name"] for g in groups):
            continue
        is_up = r in upcoming
        if is_up and parse_iso(r["net"]) - now > timedelta(hours=72):
            continue
        sg = src.get("supgp_" + r["file"], SUPGP.format(r["file"]))
        if not isinstance(sg, list) or not sg:
            continue
        inc = float(np.median([o["INCLINATION"] for o in sg]))
        r["inc"], r["inc_source"] = round(inc, 1), "spacex"
        groups.append({"key": r["name"], "intdes": None, "launch_dt": parse_iso(r["net"]),
                       "inc": inc, "height_km": round(height_km(sg[0]["MEAN_MOTION"])),
                       "sats": [sat_from_omm(o) for o in sg],
                       "single_stack": len(sg) <= 3, "forecast": is_up})

    for r in upcoming + recent:
        if r["inc"] is None and r["group"] in GROUP_INC:
            r["inc"], r["inc_source"] = GROUP_INC[r["group"]], "group_map"
        r["verdict"], r["verdict_pl"] = visibility_verdict(r["inc"])

    # 4. przeloty
    t0 = ts.from_datetime(now - timedelta(days=PAST_DAYS))
    t1 = ts.from_datetime(now + timedelta(days=PREDICT_DAYS))
    passes, season = {}, {}
    for key, c in CITIES.items():
        topos = wgs84.latlon(c["lat"], c["lon"], 100)
        ev = []
        for g in groups:
            if g["inc"] < 38:
                continue
            ev += train_events(g, topos, t0, t1, now)
        ev.sort(key=lambda e: e["start"])
        passes[key] = ev
        d0 = now.astimezone(TZ_PL).date()
        season[key] = [light_windows(topos, d0 + timedelta(days=k)) for k in range(2)]

    # 5. liczba satelitów (raz na dobę)
    count = None
    if a.prev and os.path.exists(a.prev):
        try:
            old = json.load(open(a.prev, encoding="utf-8")).get("constellation")
            if old and now - parse_iso(old["as_of"]) < timedelta(hours=24):
                count = old
        except Exception:
            pass
    if count is None:
        allsl = src.get("gp_starlink", GP_STARLINK)
        if isinstance(allsl, list) and allsl:
            count = {"count": len(allsl), "as_of": now.strftime("%Y-%m-%dT%H:%M:%SZ"), "source": "CelesTrak"}

    def clean(r):
        return {k: r.get(k) for k in ("name", "net", "net_precision", "status", "status_name", "site", "group", "inc",
                                      "inc_source", "verdict", "verdict_pl")}

    out = {
        "version": VERSION,
        "generated_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "params": {"min_alt": MIN_ALT, "list_min_max_alt": LIST_MIN_MAX_ALT, "sun_max": SUN_MAX,
                   "max_age_days": MAX_AGE_DAYS, "predict_days": PREDICT_DAYS},
        "cities": CITIES,
        "launches_upcoming": [clean(r) for r in sorted(upcoming, key=lambda r: r["net"])],
        "launches_recent": [clean(r) for r in sorted(recent, key=lambda r: r["net"], reverse=True)],
        "groups": [{"name": g["key"], "intdes": g["intdes"], "inc": round(g["inc"], 1),
                    "height_km": g["height_km"], "n": len(g["sats"]), "forecast": g["forecast"]}
                   for g in groups],
        "passes": passes,
        "season": season,
        "constellation": count,
        "log": src.log,
    }
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, separators=(",", ":"))
    n = sum(len(v) for v in passes.values())
    print(f"OK {a.out}: grup {len(groups)}, startów nadch. {len(upcoming)}, pociągów (10 miast) {n}, "
          f"log: {src.log}")


if __name__ == "__main__":
    main()
