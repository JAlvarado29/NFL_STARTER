#!/usr/bin/env python3
import os, sys, httpx
BASE=os.environ.get("NFL_API_BASE","http://127.0.0.1:8000")
def main():
    url=f"{BASE}/predict/current"
    try:
        with httpx.Client(timeout=60.0) as cx:
            r=cx.get(url); r.raise_for_status(); data=r.json()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr); sys.exit(1)
    week=data.get("week"); games=data.get("games", [])
    print(f"\nNFL Week {week} — model picks\n")
    for g in games:
        label=f"{g['away']} at {g['home']}"; pr=g.get("prediction")
        if pr: print(f"- {label}:  {pr['pick']}  ({pr['prob']*100:.1f}%)")
        else: print(f"- {label}:  no prediction (missing features)")
        if g.get("final"):
            fin=g["final"]; corr=g.get("correct"); mark="✅" if corr else ("❌" if corr is not None else "")
            print(f"    Final: {fin['score']} — Winner: {fin['winner']} {mark}")
    print()
if __name__=='__main__': main()
