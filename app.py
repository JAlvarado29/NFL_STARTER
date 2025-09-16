from __future__ import annotations
import os
from typing import Dict, List
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# --- External dependency check ---
try:
    import nfl_data_py as nfl
except Exception as e:
    raise RuntimeError("nfl_data_py is required. Install with: pip install nfl_data_py") from e

# --- FastAPI app setup ---
app = FastAPI(title="NFL Predictor API (scratch)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Season setting ---
SEASON = int(os.getenv("NFL_SEASON", "2025"))

# --- Team Abbreviations Map ---
TEAM_NAME: Dict[str, str] = {
    "ARI": "Arizona Cardinals","SEA": "Seattle Seahawks","SFO": "San Francisco 49ers","SF": "San Francisco 49ers",
    "LAR": "Los Angeles Rams","LAC": "Los Angeles Chargers","KAN": "Kansas City Chiefs","KC": "Kansas City Chiefs",
    "DEN": "Denver Broncos","LVR": "Las Vegas Raiders","LV": "Las Vegas Raiders",
    "GNB": "Green Bay Packers","GB": "Green Bay Packers","DET": "Detroit Lions","MIN": "Minnesota Vikings","CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals","CLE": "Cleveland Browns","PIT": "Pittsburgh Steelers","BAL": "Baltimore Ravens",
    "DAL": "Dallas Cowboys","PHI": "Philadelphia Eagles","NYG": "New York Giants","WAS": "Washington Commanders","WSH":"Washington Commanders",
    "BUF": "Buffalo Bills","NYJ": "New York Jets","MIA": "Miami Dolphins","NWE": "New England Patriots","NE":"New England Patriots",
    "ATL": "Atlanta Falcons","CAR": "Carolina Panthers","TAM":"Tampa Bay Buccaneers","TB":"Tampa Bay Buccaneers",
    "NOR":"New Orleans Saints","NO":"New Orleans Saints","HOU":"Houston Texans","IND":"Indianapolis Colts",
    "JAX":"Jacksonville Jaguars","JAC":"Jacksonville Jaguars","TEN":"Tennessee Titans",
}

def _to_full(abbr: str) -> str:
    """Convert team abbreviation to full name."""
    a = (abbr or "").strip().upper()
    return TEAM_NAME.get(a, abbr)

# --- Schedule Loader ---
def load_schedule(season: int) -> pd.DataFrame:
    """Load NFL schedule for a given season using nfl_data_py."""
    df = nfl.import_schedules([season]).copy()

    # Filter for regular season games only
    if "season_type" in df.columns:
        mask = df["season_type"].astype(str).str.upper().isin(["REG","REGULAR","REGULAR_SEASON"])
        df = df.loc[mask].copy()
    elif "game_type" in df.columns:
        mask = df["game_type"].astype(str).str.upper().isin(["REG"])
        df = df.loc[mask].copy()
    elif "week" in df.columns:
        df = df[df["week"].between(1,18, inclusive="both")].copy()

    # Identify home/away columns
    home_col = "home_team" if "home_team" in df.columns else ("team_home" if "team_home" in df.columns else None)
    away_col = "away_team" if "away_team" in df.columns else ("team_away" if "team_away" in df.columns else None)

    if not home_col or not away_col:
        raise RuntimeError(f"Could not find home/away team columns. Columns: {list(df.columns)}")

    # Normalize team names
    df["home"] = df[home_col].apply(_to_full)
    df["away"] = df[away_col].apply(_to_full)

    # Keep only relevant columns
    keep = ["season","week",away_col,home_col,"away","home"]
    for c in ["home_score","away_score"]:
        if c in df.columns:
            keep.append(c)

    df = df[keep].sort_values(["week","home","away"]).reset_index(drop=True)

    # Unique game key
    df["game_key"] = df.apply(
        lambda r: f"{season}-W{int(r['week']):02d}-{r[away_col]}@{r[home_col]}", axis=1
    )
    return df

def as_week_dict(df: pd.DataFrame) -> Dict[int, List[dict]]:
    """Convert schedule DataFrame into a dictionary grouped by week."""
    out: Dict[int,List[dict]] = {}
    for wk, chunk in df.groupby("week", sort=True):
        out[int(wk)] = [{"away": r.away, "home": r.home, "game_key": r.game_key} for r in chunk.itertuples(index=False)]
    return out

# Initialize schedule once at startup
_SCHEDULE_DF = load_schedule(SEASON)
NFL_BY_WEEK = as_week_dict(_SCHEDULE_DF)

# --- Data sources ---
from data_sources import build_master_features, espn_scoreboard, predict_master_row

def _attach_actuals(week: int, rows: List[dict]) -> List[dict]:
    """Attach actual final scores (if completed) to prediction rows."""
    try:
        scores = espn_scoreboard(week=week)
    except Exception:
        scores = pd.DataFrame()

    smap = {}
    if not scores.empty:
        for r in scores.itertuples(index=False):
            if getattr(r,"home",None) and getattr(r,"away",None):
                smap[f"{r.away} at {r.home}"] = r

    for r in rows:
        s = smap.get(f"{r['away']} at {r['home']}")
        if s and getattr(s,"completed",False):
            hs, as_ = getattr(s,"home_score",None), getattr(s,"away_score",None)
            winner = (
                s.home if (hs is not None and as_ is not None and hs > as_) 
                else (s.away if (hs is not None and as_ is not None and as_ > hs) else None)
            )
            r["final"] = {
                "winner": winner,
                "score": f"{s.away}: {as_} — {s.home}: {hs}" if (hs is not None and as_ is not None) else "Final"
            }
            if r.get("prediction") and winner:
                r["correct"] = (r["prediction"]["pick"] == winner)
    return rows

# --- API Endpoints ---
@app.get("/health")
def health():
    return {"ok": True, "season": SEASON, "games": int(_SCHEDULE_DF.shape[0])}

@app.get("/weeks")
def weeks():
    return {"season": SEASON, "weeks": sorted(map(int, NFL_BY_WEEK.keys()))}

@app.get("/schedule/{week}")
def schedule(week: int):
    return {"season": SEASON, "week": week, "games": NFL_BY_WEEK.get(int(week), [])}

@app.get("/predict/week/{week}")
def predict_week(week: int, as_of: str | None = None):
    feats = build_master_features(_SCHEDULE_DF, season=SEASON, week=week, as_of=as_of)
    games = []

    for f in feats.itertuples(index=False):
        d = f._asdict()
        pred = predict_master_row(pd.Series(d))
        games.append({
            "home": d.get("home"),
            "away": d.get("away"),
            "prediction": {
                "pick": pred["pick"],
                "confidence": round(pred["prob"], 3)
            }
        })

    games = _attach_actuals(week, games)

    return {"season": SEASON, "week": week, "count": len(games), "games": games}

@app.get("/predict/current")
def predict_current():
    try:
        df = espn_scoreboard()
        wk = pd.to_numeric(df["week"], errors="coerce").dropna()
        if not wk.empty:
            return predict_week(int(wk.iloc[0]))
    except Exception:
        pass
    return predict_week(2)

@app.get("/scores/espn")
def scores_espn(season: int | None = None, dates: str | None = None, week: int | None = None):
    from data_sources import espn_scoreboard  # avoid circular import
    s = season or SEASON
    df = espn_scoreboard(season=s, dates=dates, week=week)
    return {"season": s, "count": int(df.shape[0]), "games": df.to_dict(orient="records")}
