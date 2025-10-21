# main.py — Skandináv lottó (auto-fetch, resilient parser, clean UI)

import re, itertools, random, time
from collections import Counter
from datetime import datetime, timezone
from io import StringIO
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import binom
from dateutil import parser as dtp

import requests
from bs4 import BeautifulSoup
import streamlit as st
import altair as alt

# ---------------- constants ----------------
N_NUMBERS = 35
DRAW_SIZE = 7
ARCHIVE_URL = "https://bet.szerencsejatek.hu/cmsfiles/skandi.html"

# ---------------- Altair theme ----------------
@alt.theme.register("clean", enable=True)
def _alt_theme():
    return {
        "config": {
            "view": {"stroke": "transparent"},
            "axis": {"domain": False, "grid": True, "gridColor": "#e5e7eb", "tickColor": "#e5e7eb"},
            "bar": {"cornerRadiusTopLeft": 4, "cornerRadiusTopRight": 4},
            "legend": {"orient": "bottom"},
        }
    }
alt.data_transformers.disable_max_rows()

# ---------------- helpers ----------------
def _parse_date(s: str):
    s = str(s).strip()
    if not s: return pd.NaT
    m = re.search(r"(\d{4})[.\-/](\d{2})[.\-/](\d{2})", s)
    if m:
        y, mo, d = map(int, m.groups())
        try: return datetime(y, mo, d)
        except Exception: return pd.NaT
    try: return dtp.parse(s, dayfirst=True, yearfirst=True)
    except Exception: return pd.NaT

def _ints(text: str) -> List[int]:
    return [int(x) for x in re.findall(r"\d+", str(text))]

def _get_prizes(text: str) -> dict:
    blocks = re.findall(r"(\d+)\s*db\s*([\d\s]+)\s*Ft", text)
    c = [0,0,0,0]; a = [0,0,0,0]
    if len(blocks) >= 4:
        for i in range(4):
            c[i] = int(blocks[i][0])
            a[i] = int(re.sub(r"\D", "", blocks[i][1])) if re.search(r"\d", blocks[i][1]) else 0
    return {
        "winners_7_count": c[0], "winners_7_amount_ft": a[0],
        "winners_6_count": c[1], "winners_6_amount_ft": a[1],
        "winners_5_count": c[2], "winners_5_amount_ft": a[2],
        "winners_4_count": c[3], "winners_4_amount_ft": a[3],
    }

def _row_from_text(line: str) -> Optional[Tuple[int,int,pd.Timestamp,List[int],List[int],dict]]:
    # need at least 14 small numbers and a year/week source (explicit or via date)
    nums_all = _ints(line)
    small = [x for x in nums_all if 1 <= x <= 35]
    if len(small) < 14:
        return None

    dt = _parse_date(line)
    year = None; week = None

    # try explicit year + week near the start
    head = nums_all[:6]
    for v in head:
        if year is None and 1900 <= v <= 2100:
            year = v
        elif week is None and 1 <= v <= 53:
            week = v
    # fallback from date
    if (year is None or week is None) and pd.notna(dt):
        iso = pd.Timestamp(dt).isocalendar()
        year = int(iso.year) if year is None else year
        week = int(iso.week) if week is None else week
    if year is None or week is None:
        return None

    # choose a stable 14-block: prefer the last 14 small numbers
    block = small[-14:]
    a, b = sorted(block[:7]), sorted(block[7:])
    meta = _get_prizes(line)
    return year, week, pd.Timestamp(dt) if pd.notna(dt) else pd.NaT, a, b, meta

# ---------------- fetch (resilient) ----------------
@st.cache_data(show_spinner=True, ttl=12*3600)
def fetch_skandi() -> pd.DataFrame:
    ses = requests.Session()
    headers = {"User-Agent": "Mozilla/5.0"}
    for attempt in range(5):
        try:
            r = ses.get(ARCHIVE_URL, headers=headers, timeout=25)
            r.raise_for_status()
            html = r.text
            break
        except Exception:
            if attempt == 4:
                raise
            time.sleep(0.8 * (2 ** attempt))

    soup = BeautifulSoup(html, "lxml")
    rows = []

    # Strategy A: parse visible text lines
    text = soup.get_text("\n").replace("\xa0", " ")
    for line in text.splitlines():
        tup = _row_from_text(line)
        if tup:
            y, w, dt, g, k, prize = tup
            meta = dict(year_lottery=y, week_lottery=w, draw_date=dt, **prize)
            rows.append({**meta, "draw_type":"gepi", **{f"n{i+1}": g[i] for i in range(7)}})
            rows.append({**meta, "draw_type":"kezi", **{f"n{i+1}": k[i] for i in range(7)}})

    # Strategy B: parse any tables by row text (no pandas read_html)
    if not rows:
        for table in soup.find_all("table"):
            for tr in table.find_all("tr"):
                line = " ".join(td.get_text(" ", strip=True) for td in tr.find_all(["td","th"]))
                tup = _row_from_text(line)
                if tup:
                    y, w, dt, g, k, prize = tup
                    meta = dict(year_lottery=y, week_lottery=w, draw_date=dt, **prize)
                    rows.append({**meta, "draw_type":"gepi", **{f"n{i+1}": g[i] for i in range(7)}})
                    rows.append({**meta, "draw_type":"kezi", **{f"n{i+1}": k[i] for i in range(7)}})

    if not rows:
        # Provide a short diagnostic sample in Streamlit instead of crashing
        sample = "\n".join(text.splitlines()[:30])
        st.error("Could not parse the archive. The page format changed.")
        st.text(sample)
        st.stop()

    df = pd.DataFrame(rows).drop_duplicates(subset=["year_lottery","week_lottery","draw_type"])
    # sanity
    for i in range(1,8):
        df = df[(df[f"n{i}"] >= 1) & (df[f"n{i}"] <= N_NUMBERS)]
    df["draw_date"] = pd.to_datetime(df["draw_date"], errors="coerce")
    df = df.sort_values(["year_lottery","week_lottery","draw_type"]).reset_index(drop=True)
    if df.empty:
        st.error("Parsed rows filtered to empty after sanity checks.")
        st.stop()
    return df

# ---------------- features ----------------
def add_presence(df: pd.DataFrame) -> pd.DataFrame:
    vals = df[[f"n{i}" for i in range(1,8)]].to_numpy(dtype=int)
    pres = np.zeros((len(df), N_NUMBERS), dtype=int)
    for j in range(7):
        v = vals[:, j]; mask = (v >= 1) & (v <= N_NUMBERS)
        pres[np.where(mask)[0], v[mask]-1] = 1
    for i in range(N_NUMBERS):
        df[f"has_{i+1:02d}"] = pres[:, i]
    return df

def ewma_hit_rate(df: pd.DataFrame, half_life_draws: int):
    pres = df[[f"has_{i:02d}" for i in range(1,N_NUMBERS+1)]].to_numpy(dtype=float)
    T = pres.shape[0]
    if T == 0: return np.zeros(N_NUMBERS)
    k = np.arange(T); d = (T-1) - k
    w = np.power(0.5, d / max(1, half_life_draws))[:, None]
    return (pres * w).sum(axis=0) / w.sum()

def gaps_last_seen(df: pd.DataFrame):
    idx = np.arange(len(df)); last = np.full(N_NUMBERS, np.nan, dtype=float)
    for i in range(1,N_NUMBERS+1):
        occ = idx[df[f"has_{i:02d}"] == 1]
        last[i-1] = (len(df)-1 - occ[-1]) if len(occ) else np.nan
    return last

def zscore(a: np.ndarray):
    m, s = np.nanmean(a), np.nanstd(a, ddof=1)
    return np.zeros_like(a) if not np.isfinite(s) or s == 0 else (a - m) / s

def normalize_nonneg(a: np.ndarray):
    a = np.clip(a,0,None); s = a.sum()
    return np.full_like(a, 1.0/len(a)) if s <= 0 else a/s

def weighted_sample(items, weights, k, rng: random.Random):
    items, weights = list(items), list(weights); out = []
    for _ in range(k):
        tot = sum(weights)
        idx = rng.randrange(len(items)) if tot <= 0 else next(
            i for i,_ in enumerate(weights) if sum(weights[:i+1]) >= rng.random()*tot
        )
        out.append(items[idx]); del items[idx]; del weights[idx]
    return out

def rolling_rates(df_draws: pd.DataFrame, window:int=52) -> pd.DataFrame:
    pres = df_draws[[f"has_{i:02d}" for i in range(1,N_NUMBERS+1)]].copy()
    roll = pres.rolling(window, min_periods=max(5, window//4)).mean()
    roll.index = np.arange(len(roll))
    return roll

# ---------------- UI base ----------------
st.set_page_config(page_title="Skandináv lottó", layout="wide")
st.markdown("""
<style>
section.main { max-width: 1200px; }
.card { border:1px solid var(--card-border, #e5e7eb); border-radius:12px; padding:12px; margin-bottom:12px; }
.hint { opacity:.8; font-size:.9rem; margin-bottom:6px; }
/* balls */
:root { --ball-bg:#0b1220; --ball-fg:#eef2ff; --ball-ring:#334155; }
@media (prefers-color-scheme: light) { :root { --ball-bg:#ffffff; --ball-fg:#111827; --ball-ring:#d1d5db; } }
.ball { display:inline-flex; align-items:center; justify-content:center; width:38px; height:38px; margin:4px; border-radius:50%;
  font-weight:700; font-variant-numeric: tabular-nums; background:var(--ball-bg); color:var(--ball-fg);
  border:1px solid var(--ball-ring); box-shadow:0 1px 0 rgba(0,0,0,.15), inset 0 -2px 0 rgba(0,0,0,.06); }
.ball.accent { background:linear-gradient(145deg,#1d4ed8,#2563eb); color:#fff; border-color:#1e40af; }
.kpi { text-align:center; } .kpi h3{margin:.2rem 0 .3rem 0; font-weight:600;} .kpi .val{font-size:1.3rem; font-weight:800;}
</style>
""", unsafe_allow_html=True)

# ---------------- fetch ----------------
with st.spinner("Loading the full archive…"):
    df_all = fetch_skandi()
df_all = add_presence(df_all)
df_gepi = df_all[df_all["draw_type"]=="gepi"].reset_index(drop=True)
df_kezi = df_all[df_all["draw_type"]=="kezi"].reset_index(drop=True)

# ---------------- sidebar ----------------
with st.sidebar:
    st.header("Pick settings")
    recent = st.slider("Favor recent weeks", 0, 100, 60, step=5)
    with st.expander("More options", expanded=False):
        popular = st.slider("Favor often-drawn", 0, 100, 25, step=5)
        overdue = st.slider("Favor long-absent", 0, 100, 15, step=5)
        half_life = st.number_input("Recent window (draws)", 1, 200, 52, step=1)
        samples = st.number_input("Extra ideas", 0, 8, 2, step=1)
        seed = st.number_input("Random seed", value=42, step=1)
        if st.button("Refresh data"): fetch_skandi.clear(); st.rerun()
    st.caption(datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"))

# ---------------- signals ----------------
counts_all = df_all[[f"has_{i:02d}" for i in range(1,N_NUMBERS+1)]].sum().to_numpy(dtype=float)
base_rate = counts_all / max(1, len(df_all))
ew = ewma_hit_rate(df_all, int(half_life))
gap_z = zscore(gaps_last_seen(df_all))
rec_z = zscore(ew); base_z = zscore(base_rate)
w_sum = max(1e-9, recent + popular + overdue)
WR, WB, WG = recent/w_sum, popular/w_sum, overdue/w_sum
score = WR*rec_z + WB*base_z + WG*gap_z
weights = normalize_nonneg(score - score.min() + 1e-9)

# ---------------- picks ----------------
rng = random.Random(int(st.session_state.get("_seed", int(seed))))
top_balanced = list(np.argsort(-score)[:DRAW_SIZE] + 1)
top_recent   = list(np.argsort(-ew)[:DRAW_SIZE] + 1)
top_overdue  = list(np.argsort(-gap_z)[:DRAW_SIZE] + 1)
top_popular  = list(np.argsort(-base_rate)[:DRAW_SIZE] + 1)
ideas = [sorted(weighted_sample(range(1,N_NUMBERS+1), weights, DRAW_SIZE, rng)) for _ in range(int(samples))]

pair_all = Counter()
for row in df_all[[f"n{i}" for i in range(1,8)]].to_numpy():
    for a,b in itertools.combinations(sorted(row),2):
        pair_all[(int(a),int(b))] += 1

def balls_html(nums, accent=False):
    cls = "ball accent" if accent else "ball"
    return "".join(f"<span class='{cls}'>{n:02d}</span>" for n in sorted(nums))
def line_card(title, nums, accent=False):
    st.markdown(f"<div class='card'><div class='hint'>{title}</div>{balls_html(nums, accent)}</div>", unsafe_allow_html=True)

# ---------------- KPIs ----------------
latest_dt = df_all.dropna(subset=["draw_date"])["draw_date"].max()
k1,k2,k3,k4 = st.columns(4)
with k1: st.markdown(f"<div class='kpi'><h3>years</h3><div class='val'>{df_all['year_lottery'].min()}–{df_all['year_lottery'].max()}</div></div>", unsafe_allow_html=True)
with k2: st.markdown(f"<div class='kpi'><h3>weeks</h3><div class='val'>{df_all[['year_lottery','week_lottery']].drop_duplicates().shape[0]:,}</div></div>", unsafe_allow_html=True)
with k3: st.markdown(f"<div class='kpi'><h3>draws</h3><div class='val'>{len(df_all):,}</div></div>", unsafe_allow_html=True)
with k4: st.markdown(f"<div class='kpi'><h3>latest</h3><div class='val'>{latest_dt.date().isoformat() if pd.notna(latest_dt) else '-'}</div></div>", unsafe_allow_html=True)

# ---------------- header ----------------
left, right = st.columns([1.25, 1])
with left:
    st.subheader("Your lines")
    line_card("Balanced", top_balanced, accent=True)
    c1,c2 = st.columns(2)
    with c1: line_card("Recent hot", top_recent)
    with c2: line_card("Long-absent", top_overdue)
    line_card("Often drawn", top_popular)
    if ideas:
        with st.expander("More ideas"):
            for s in ideas: line_card(" ", s)
    pick_text = "\n".join([
        f"Balanced: {' '.join(f'{n:02d}' for n in sorted(top_balanced))}",
        f"Recent:   {' '.join(f'{n:02d}' for n in sorted(top_recent))}",
        f"Overdue:  {' '.join(f'{n:02d}' for n in sorted(top_overdue))}",
        f"Popular:  {' '.join(f'{n:02d}' for n in sorted(top_popular))}",
    ] + [f"Idea {i+1}: {' '.join(f'{n:02d}' for n in s)}" for i,s in enumerate(ideas)])
    st.download_button("Download picks (.txt)", data=pick_text, file_name="skandi_picks.txt", type="secondary")

with right:
    st.subheader("Top numbers now")
    ranks = pd.DataFrame({"number": np.arange(1,N_NUMBERS+1), "priority": score}).sort_values("priority", ascending=False).head(12)
    chart = alt.Chart(ranks).mark_bar().encode(
        x=alt.X("number:O", title=""),
        y=alt.Y("priority:Q", title=""),
        tooltip=[alt.Tooltip("number:O", title="number"), alt.Tooltip("priority:Q", format=".2f")],
    ).properties(height=240)
    st.altair_chart(chart, use_container_width=True)
    st.caption("Blend of recent form, long-run, and time since last seen.")

# ---------------- tabs ----------------
tab_explore, tab_pairs, tab_patterns, tab_inspect, tab_recent = st.tabs(
    ["Numbers", "Pairs", "Patterns", "Inspector", "Last 12 weeks"]
)
with tab_explore:
    view = st.radio("Show", ["All", "Gépi", "Kézi"], index=0, horizontal=True)
    df_sel = {"All": df_all, "Gépi": df_gepi, "Kézi": df_kezi}[view]
    counts = df_sel[[f"has_{i:02d}" for i in range(1,N_NUMBERS+1)]].sum().to_numpy(dtype=int)
    freq = pd.DataFrame({"number": np.arange(1,N_NUMBERS+1), "hits": counts})
    ch = alt.Chart(freq).mark_bar().encode(
        x=alt.X("number:O", title="number"),
        y=alt.Y("hits:Q", title="times", scale=alt.Scale(zero=True)),
        tooltip=["number","hits"]
    ).properties(height=330, title=f"How often each number appeared ({view.lower()})")
    st.altair_chart(ch, use_container_width=True)
    st.caption("All years combined.")

with tab_pairs:
    M = np.zeros((N_NUMBERS, N_NUMBERS), dtype=int)
    mat = df_all[[f"n{i}" for i in range(1, 8)]].to_numpy()
    for row in mat:
        for a, b in itertools.combinations(sorted(row), 2):
            M[a - 1, b - 1] += 1
            M[b - 1, a - 1] += 1

    hm = pd.DataFrame(
        [(i + 1, j + 1, int(M[i, j])) for i in range(N_NUMBERS) for j in range(N_NUMBERS)],
        columns=["x", "y", "together"],
    )

    heat = alt.Chart(hm).mark_rect().encode(
        x=alt.X("x:O", title=""),
        y=alt.Y("y:O", title=""),
        color=alt.Color("together:Q", title="together", scale=alt.Scale(scheme="blues")),
        tooltip=[
            alt.Tooltip("x:O", title="A"),
            alt.Tooltip("y:O", title="B"),
            alt.Tooltip("together:Q", title="together"),
        ],
    ).properties(height=420, title="Numbers that appear together")

    st.altair_chart(heat, use_container_width=True)

    pairs_list = Counter({(i + 1, j + 1): M[i, j] for i in range(N_NUMBERS) for j in range(i + 1, N_NUMBERS)})
    top_pairs = pd.DataFrame(
        [{"pair": f"{a:02d}-{b:02d}", "times": c} for (a, b), c in pairs_list.most_common(15)]
    )
    st.dataframe(top_pairs, height=320)

with tab_patterns:
    nums = df_all[[f"n{i}" for i in range(1,8)]]
    sums = nums.sum(axis=1).rename("total").to_frame()
    st.altair_chart(
        alt.Chart(sums).mark_bar().encode(
            x=alt.X("total:Q", bin=alt.Bin(maxbins=30), title="total of the 7 numbers"),
            y=alt.Y("count()", title="draws", scale=alt.Scale(zero=True)),
            tooltip=[alt.Tooltip("count()", title="draws")]).properties(height=240, title="Total of the 7 numbers"),
        use_container_width=True
    )
    odd_counts = nums.apply(lambda r: int(np.sum(np.array(r) % 2 == 1)), axis=1)
    obs = odd_counts.value_counts().sort_index()
    support = list(range(0,DRAW_SIZE+1))
    exp = [binom.pmf(k, DRAW_SIZE, 0.5)*len(df_all) for k in support]
    df_odd = pd.DataFrame({"odd": support, "observed": [int(obs.get(k,0)) for k in support], "expected": exp})
    st.altair_chart(
        alt.Chart(df_odd.melt("odd", var_name="type", value_name="draws")).mark_bar().encode(
            x=alt.X("odd:O", title="odd numbers in a draw"),
            y=alt.Y("draws:Q", title="draws", scale=alt.Scale(zero=True)),
            color=alt.Color("type:N", title=""),
            tooltip=["odd","type", alt.Tooltip("draws:Q", format=".1f")]).properties(height=240, title="Odd vs chance"),
        use_container_width=True
    )
    spread = (nums.max(axis=1) - nums.min(axis=1)).rename("range").to_frame()
    st.altair_chart(
        alt.Chart(spread).mark_bar().encode(
            x=alt.X("range:Q", bin=alt.Bin(maxbins=30), title="range (max − min)"),
            y=alt.Y("count()", title="draws", scale=alt.Scale(zero=True)),
            tooltip=[alt.Tooltip("count()", title="draws")]).properties(height=240, title="How spread the 7 numbers were"),
        use_container_width=True
    )

with tab_inspect:
    st.write("Pick a number to see its story.")
    n = st.slider("Number", 1, N_NUMBERS, 7)
    total_hits = int(counts_all[n-1]); share = total_hits / max(1, len(df_all))
    last_gap = gaps_last_seen(df_all)[n-1]; recent_rate = ew[n-1]
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("times drawn", f"{total_hits}")
    c2.metric("share of draws", f"{share:.3f}")
    c3.metric("last seen (draws)", "-" if np.isnan(last_gap) else f"{int(last_gap)}")
    c4.metric("recent rate", f"{recent_rate:.3f}")
    roll = rolling_rates(df_all, window=52)
    ts = pd.DataFrame({"t": roll.index, "rate": roll[f"has_{n:02d}"]})
    st.altair_chart(
        alt.Chart(ts).mark_line().encode(
            x=alt.X("t:Q", title="draw index"),
            y=alt.Y("rate:Q", title="rate (last 52 draws)"),
            tooltip=[alt.Tooltip("t:Q", title="draw"), alt.Tooltip("rate:Q", format=".3f")]
        ).properties(height=240, title=f"Number {n:02d} — recent form"),
        use_container_width=True
    )

with tab_recent:
    st.write("Last 12 weeks, gépi vs kézi")
    weeks = df_all[["year_lottery","week_lottery"]].drop_duplicates().sort_values(
        ["year_lottery","week_lottery"], ascending=[False, False]).head(12)
    show = weeks.merge(df_gepi, on=["year_lottery","week_lottery"], how="left").merge(
        df_kezi, on=["year_lottery","week_lottery"], how="left", suffixes=("_g","_k"))
    def balls(nums): return "".join(f"<span class='ball'>{n:02d}</span>" for n in nums)
    for _, r in show.iterrows():
        st.markdown(f"**{int(r['year_lottery'])}/{int(r['week_lottery'])}**")
        two = st.columns(2)
        with two[0]:
            st.markdown(f"<div class='card'><div class='hint'>gépi</div>{balls([int(r.get(f'n{i}_g',0)) for i in range(1,8)])}</div>", unsafe_allow_html=True)
        with two[1]:
            st.markdown(f"<div class='card'><div class='hint'>kézi</div>{balls([int(r.get(f'n{i}_k',0)) for i in range(1,8)])}</div>", unsafe_allow_html=True)

st.caption("Source: Szerencsejáték Zrt archive. Past results only; odds unchanged.")