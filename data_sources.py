from __future__ import annotations
import datetime as dt, os, re
from functools import lru_cache
from typing import Dict, List, Optional
import httpx, pandas as pd
from bs4 import BeautifulSoup, Comment

USER_AGENT = os.getenv("HTTP_USER_AGENT","NFLPredictor/1.0 (+https://example.com)")
TIMEOUT = float(os.getenv("HTTP_TIMEOUT","30"))
SEASON = int(os.getenv("NFL_SEASON","2025"))

TEAM_CANON: Dict[str,str] = {
 "Buffalo Bills":"Buffalo Bills","Miami Dolphins":"Miami Dolphins","New England Patriots":"New England Patriots","New York Jets":"New York Jets",
 "Baltimore Ravens":"Baltimore Ravens","Cincinnati Bengals":"Cincinnati Bengals","Cleveland Browns":"Cleveland Browns","Pittsburgh Steelers":"Pittsburgh Steelers",
 "Houston Texans":"Houston Texans","Indianapolis Colts":"Indianapolis Colts","Jacksonville Jaguars":"Jacksonville Jaguars","Tennessee Titans":"Tennessee Titans",
 "Denver Broncos":"Denver Broncos","Kansas City Chiefs":"Kansas City Chiefs","Las Vegas Raiders":"Las Vegas Raiders","Los Angeles Chargers":"Los Angeles Chargers",
 "Dallas Cowboys":"Dallas Cowboys","New York Giants":"New York Giants","Philadelphia Eagles":"Philadelphia Eagles","Washington Commanders":"Washington Commanders",
 "Chicago Bears":"Chicago Bears","Detroit Lions":"Detroit Lions","Green Bay Packers":"Green Bay Packers","Minnesota Vikings":"Minnesota Vikings",
 "Atlanta Falcons":"Atlanta Falcons","Carolina Panthers":"Carolina Panthers","New Orleans Saints":"New Orleans Saints","Tampa Bay Buccaneers":"Tampa Bay Buccaneers",
 "Arizona Cardinals":"Arizona Cardinals","Los Angeles Rams":"Los Angeles Rams","San Francisco 49ers":"San Francisco 49ers","Seattle Seahawks":"Seattle Seahawks",
}
SUMER_OFFENSE_URL="https://sumersports.com/teams/offensive/"
SUMER_DEFENSE_URL="https://sumersports.com/teams/defensive/"
ESPN_SCOREBOARD="https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"

def _http_get(url: str, params: Optional[dict]=None) -> str:
    with httpx.Client(headers={"User-Agent": USER_AGENT}) as cx:
        r=cx.get(url, params=params, timeout=TIMEOUT); r.raise_for_status(); return r.text

def _http_get_json(url: str, params: Optional[dict]=None) -> dict:
    with httpx.Client(headers={"User-Agent": USER_AGENT}) as cx:
        r=cx.get(url, params=params, timeout=TIMEOUT); r.raise_for_status(); return r.json()

def _choose_sumer_table(html: str) -> pd.DataFrame:
    try: tables = pd.read_html(html)
    except ValueError: tables = []
    candidates=[t for t in tables if any(str(c).lower().startswith("team") for c in t.columns)]
    if candidates: return max(candidates, key=lambda t: t.shape[0]*t.shape[1]).copy()
    soup=BeautifulSoup(html,"html.parser"); cards=[]
    for img in soup.select("img[alt='logo'], img[alt*='logo']"):
        card=img.find_parent(); 
        if not card: continue
        text=card.get_text(" ", strip=True)
        m=re.search(r"([A-Z][A-Za-z .'-]+)\s+20\d{2}", text)
        if not m: continue
        team=m.group(1); numbers=re.findall(r"[-+]?[0-9]*\.?[0-9]+%?", text)
        cards.append({"Team": team, "raw": numbers})
    if cards: return pd.DataFrame(cards)
    raise RuntimeError("Could not parse SumerSports page.")

def _normalize_sumer_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename={"Team":"team","Season":"season","EPA":"epa","Success %":"success_pct","EPA/Pass":"epa_per_pass","EPA/Rush":"epa_per_rush",
            "Pass Yards":"pass_yards","Comp %":"comp_pct","Pass TD":"pass_td","Rush Yards":"rush_yards","Rush TD":"rush_td",
            "ADoT":"adot","Sack %":"sack_pct","Scramble %":"scramble_pct","Int %":"int_pct","Air EPA/Att":"air_epa_per_att","YAC":"yac"}
    out=df.rename(columns=rename)
    for c in ["season","epa","success_pct","epa_per_pass","epa_per_rush","pass_yards","comp_pct","pass_td","rush_yards","rush_td","adot","sack_pct","scramble_pct","int_pct","air_epa_per_att","yac"]:
        if c in out.columns: out[c]=pd.to_numeric(out[c].astype(str).str.replace("%","",regex=False), errors="coerce")
    if "team" in out.columns: out["team"]=out["team"].map(TEAM_CANON).fillna(out["team"])
    return out

@lru_cache(maxsize=4)
def fetch_sumer_offense(season: int=SEASON) -> pd.DataFrame:
    html=_http_get(SUMER_OFFENSE_URL); df=_choose_sumer_table(html); df=_normalize_sumer_columns(df)
    if "season" in df.columns: df=df[df["season"].astype("Int64")==season].copy()
    df["source"]="sumer_offense"; return df.reset_index(drop=True)

@lru_cache(maxsize=4)
def fetch_sumer_defense(season: int=SEASON) -> pd.DataFrame:
    html=_http_get(SUMER_DEFENSE_URL); df=_choose_sumer_table(html); df=_normalize_sumer_columns(df)
    if "season" in df.columns: df=df[df["season"].astype("Int64")==season].copy()
    df["source"]="sumer_defense"; return df.reset_index(drop=True)

TR_SLUGS={"points-per-game":"tr_ppg","yards-per-play":"tr_ypp","third-down-conversion-pct":"tr_3d_pct","red-zone-scoring-pct-td-only":"tr_rz_td_pct",
          "turnover-margin-per-game":"tr_to_margin","sacks-per-game":"tr_sacks_pg","yards-per-pass-attempt":"tr_ypa","yards-per-rush-attempt":"tr_ypr"}

def _pick_numeric_column(df: pd.DataFrame, season: int) -> str:
    cols=list(df.columns)
    for c in cols:
        if str(c).strip()==str(season): return c
    numeric=[(c, pd.to_numeric(df[c], errors="coerce").notna().sum()) for c in cols]
    numeric=[x for x in numeric if x[1]>=20]
    numeric.sort(key=lambda x:x[1])
    return numeric[-1][0] if numeric else cols[-1]

def fetch_teamrankings_bundle(season: int, as_of: Optional[str]=None) -> pd.DataFrame:
    date=as_of or dt.date.today().isoformat(); frames=[]
    for slug, outcol in TR_SLUGS.items():
        url=f"https://www.teamrankings.com/nfl/stat/{slug}?date={date}"
        html=_http_get(url); tables=pd.read_html(html)
        if not tables: continue
        df0=tables[0]
        if "Team" not in df0.columns: df0=df0.rename(columns={df0.columns[0]:"Team"})
        val_col=_pick_numeric_column(df0, season)
        df=pd.DataFrame({"team": df0["Team"].map(TEAM_CANON).fillna(df0["Team"]), outcol: pd.to_numeric(df0[val_col], errors="coerce")})
        frames.append(df)
    if not frames: return pd.DataFrame(columns=["team"])
    out=frames[0]
    for f in frames[1:]: out=out.merge(f, on="team", how="outer")
    out["source_tr"]="teamrankings"; return out

def _strip_html_comments(html: str) -> str:
    soup=BeautifulSoup(html,"lxml")
    for c in soup.find_all(string=lambda t: isinstance(t, Comment)): c.replace_with(c)
    return str(soup)

def fetch_pfr_drives(season: int) -> pd.DataFrame:
    url=f"https://www.pro-football-reference.com/years/{season}/drives.htm"
    html=_http_get(url); html2=_strip_html_comments(html); tables=pd.read_html(html2)
    if not tables: return pd.DataFrame(columns=["team"])
    best=None
    for t in tables:
        if "Team" in t.columns: best=t if best is None else (t if t.shape[1]>best.shape[1] else best)
    if best is None: return pd.DataFrame(columns=["team"])
    df=best.rename(columns={"Team":"team"}).copy()
    for col in list(df.columns):
        if col=="team": continue
        if any(k in str(col).lower() for k in ["pts","yd","score","to%"]):
            df[col]=pd.to_numeric(df[col].astype(str).str.replace("%","",regex=False), errors="coerce")
    df["team"]=df["team"].map(TEAM_CANON).fillna(df["team"])
    return df

def espn_scoreboard(season: int=SEASON, dates: Optional[str]=None, week: Optional[int]=None) -> pd.DataFrame:
    params={}; 
    if dates: params["dates"]=dates
    with httpx.Client(headers={"User-Agent": USER_AGENT}) as cx:
        r=cx.get(ESPN_SCOREBOARD, params=params, timeout=TIMEOUT); r.raise_for_status(); data=r.json()
    records=[]
    for ev in data.get("events", []):
        comp=ev.get("competitions",[{}])[0]; season_info=ev.get("season",{})
        wk=comp.get("week",{}).get("number") or season_info.get("week")
        if week and wk and int(wk)!=int(week): continue
        home=away=home_score=away_score=None
        for t in comp.get("competitors", []):
            name=t.get("team",{}).get("displayName"); score=t.get("score")
            if t.get("homeAway")=="home": home=name; home_score=int(score) if score is not None else None
            else: away=name; away_score=int(score) if score is not None else None
        status=comp.get("status",{}).get("type",{}).get("name"); completed=comp.get("status",{}).get("type",{}).get("completed")
        records.append({"season": season_info.get("year",season), "week": wk, "home": TEAM_CANON.get(home,home), "away": TEAM_CANON.get(away,away),
                        "home_score": home_score, "away_score": away_score, "status": status, "completed": completed, "espn_game_id": ev.get("id"), "start": ev.get("date")})
    return pd.DataFrame.from_records(records)

def build_team_strengths(season: int=SEASON) -> pd.DataFrame:
    off=fetch_sumer_offense(season); defs=fetch_sumer_defense(season)
    keep_off=[c for c in off.columns if c in {"team","epa","success_pct","epa_per_pass","epa_per_rush"}]
    keep_def=[c for c in defs.columns if c in {"team","epa","success_pct","epa_per_pass","epa_per_rush"}]
    off2=off[keep_off].rename(columns={"epa":"off_epa","success_pct":"off_success_pct","epa_per_pass":"off_epa_pass","epa_per_rush":"off_epa_rush"})
    def2=defs[keep_def].rename(columns={"epa":"def_epa","success_pct":"def_success_pct","epa_per_pass":"def_epa_pass","epa_per_rush":"def_epa_rush"})
    return pd.merge(off2, def2, on="team", how="inner")

def build_week_matchups(schedule_df: pd.DataFrame, season: int=SEASON, week: int=1) -> pd.DataFrame:
    strengths=build_team_strengths(season); df=schedule_df.copy(); df=df[df["week"]==week].copy()
    df=df.merge(strengths.add_prefix("away_"), left_on="away", right_on="away_team", how="left")
    df=df.merge(strengths.add_prefix("home_"), left_on="home", right_on="home_team", how="left")
    for c in ["away_team","home_team"]:
        if c in df.columns: df.drop(columns=[c], inplace=True)
    return df

def build_master_features(schedule_df: pd.DataFrame, season: int, week: int, as_of: Optional[str]=None) -> pd.DataFrame:
    base=build_week_matchups(schedule_df, season=season, week=week)
    try:
        tr=fetch_teamrankings_bundle(season=season, as_of=as_of)
        base=base.merge(tr.add_prefix("away_"), left_on="away", right_on="away_team", how="left"); base.drop(columns=[c for c in base.columns if c.endswith("_team")], inplace=True, errors="ignore")
        base=base.merge(tr.add_prefix("home_"), left_on="home", right_on="home_team", how="left"); base.drop(columns=[c for c in base.columns if c.endswith("_team")], inplace=True, errors="ignore")
    except Exception: pass
    try:
        pfr=fetch_pfr_drives(season=season)
        base=base.merge(pfr.add_prefix("away_"), left_on="away", right_on="away_team", how="left"); base.drop(columns=[c for c in base.columns if c.endswith("_team")], inplace=True, errors="ignore")
        base=base.merge(pfr.add_prefix("home_"), left_on="home", right_on="home_team", how="left"); base.drop(columns=[c for c in base.columns if c.endswith("_team")], inplace=True, errors="ignore")
    except Exception: pass
    def num(s): return pd.to_numeric(s, errors="coerce")
    base["edge_epa"]=(num(base.get("home_off_epa"))-num(base.get("away_def_epa")))-(num(base.get("away_off_epa"))-num(base.get("home_def_epa")))
    if "home_tr_ypp" in base.columns and "away_tr_ypp" in base.columns: base["edge_tr_ypp"]=num(base.get("home_tr_ypp"))-num(base.get("away_tr_ypp"))
    else: base["edge_tr_ypp"]=pd.NA
    hcols=[c for c in base.columns if c.startswith("home_pfr") and "pts" in c.lower()]; acols=[c for c in base.columns if c.startswith("away_pfr") and "pts" in c.lower()]
    if hcols and acols: base["edge_pfr_ppd"]=num(base[hcols[0]])-num(base[acols[0]])
    else: base["edge_pfr_ppd"]=pd.NA
    return base

from math import exp
_HFA=0.015; _W_EPA=1.0; _W_TR_YPP=0.60; _W_PFR_PPD=0.75; _SCALE=8.0
def _sigmoid(x: float) -> float: return 1.0/(1.0+exp(-x))
def predict_master_row(r: pd.Series):
    def safe(x):
        try: return float(x) if pd.notna(x) else None
        except Exception: return None
    score=0.0
    epa=safe(r.get("edge_epa")); ypp=safe(r.get("edge_tr_ypp")); ppd=safe(r.get("edge_pfr_ppd"))
    if epa is not None: score += _W_EPA*epa
    if ypp is not None: score += _W_TR_YPP*ypp
    if ppd is not None: score += _W_PFR_PPD*ppd
    score += _HFA
    p_home=_sigmoid(score*_SCALE)
    return {"pick": r.get("home"), "prob": float(p_home)} if p_home>=0.5 else {"pick": r.get("away"), "prob": float(1.0-p_home)}
