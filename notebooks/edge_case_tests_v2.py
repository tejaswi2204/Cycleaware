"""
CycleAware — Automated Edge Case Testing Script v2
===================================================
Fixes applied from v1 analysis:
1. Group 3 — Spotting rate tests now have consistent history
2. Group 4 — Gap tests now force spotting=True to actually test Model 2
3. Group 5 — Contradictory inputs redesigned for clarity

Run this in Jupyter Notebook after starting the backend:
    uvicorn app:app --reload

Then in a new cell:
    exec(open('edge_case_tests_v2.py').read())
"""

import requests
import pandas as pd

API = 'http://127.0.0.1:8000'

# ── Default payload — average neutral user ───────────────────────────────────
def default_payload():
    return {
        "age": 25, "bmi": 22.0, "cycle_length": 28,
        "stress_level": 5, "sleep_quality": 7, "pain_level": 4,
        "mood_score": 6, "hydration_level": 6,
        "prev_cycle_gap": 3, "consecutive_spotting_months": 2,
        "rolling_avg_stress_3m": 5, "cycle_length_deviation": 0.0,
        "gap_variability": 0.5, "personal_spotting_rate": 0.5,
        "personal_base_gap": 3, "month_number": 5,
        "travel_disruption": 0, "recent_illness": 0,
        "prev_cycle_had_spotting": 1, "two_months_ago_spotting": 1,
        "activity_level": 1, "stress_trend": 1,
        "bmi_category": 1, "age_group": 2, "flow_intensity": 1,
        "season_autumn": 0, "season_monsoon": 0,
        "season_summer": 1, "season_winter": 0,
        "condition_PCOS": 0, "condition_endometriosis": 0,
        "condition_none": 1, "condition_thyroid": 0,
        "contraceptive_IUD": 0, "contraceptive_implant": 0,
        "contraceptive_none": 1, "contraceptive_pill": 0,
    }

# ── Strong spotting base — used for Group 4 gap tests ────────────────────────
def spotting_base():
    """Base payload that strongly predicts spotting so Model 2 can run."""
    p = default_payload()
    p.update({
        "personal_spotting_rate": 1.0,
        "prev_cycle_had_spotting": 1,
        "two_months_ago_spotting": 1,
        "consecutive_spotting_months": 3,
        "stress_level": 8,
        "stress_trend": 2,
    })
    return p

# ── Helper: run one test ─────────────────────────────────────────────────────
def run_test(name, overrides, base="default",
             expect_spotting=None, expect_gap_direction=None, note=""):
    payload = spotting_base() if base == "spotting" else default_payload()
    payload.update(overrides)
    try:
        r = requests.post(f"{API}/predict", json=payload, timeout=10)
        if r.status_code != 200:
            return {"Test": name, "Status": "API ERROR",
                    "Spotting %": "-", "Predicted": "-",
                    "Gap": "-", "Pass": "ERROR", "Note": note}
        d = r.json()
        pct       = round(d["spotting_probability"] * 100, 1)
        predicted = "Yes" if d["spotting_predicted"] else "No"
        gap       = d["gap_days_predicted"] if d["gap_days_predicted"] else "N/A"

        passed = "—"
        if expect_spotting is not None:
            passed = "PASS" if (d["spotting_predicted"] == expect_spotting) else "FAIL"
        if expect_gap_direction == "low" and isinstance(gap, int):
            passed = "PASS" if gap <= 2 else "FAIL"
        if expect_gap_direction == "high" and isinstance(gap, int):
            passed = "PASS" if gap >= 4 else "FAIL"
        if expect_gap_direction == "mid" and isinstance(gap, int):
            passed = "PASS" if 2 <= gap <= 4 else "FAIL"

        return {
            "Test": name, "Status": "OK",
            "Spotting %": f"{pct}%", "Predicted": predicted,
            "Gap": gap, "Pass": passed, "Note": note
        }
    except Exception as e:
        return {"Test": name, "Status": f"FAILED — {str(e)}",
                "Spotting %": "-", "Predicted": "-",
                "Gap": "-", "Pass": "ERROR", "Note": note}

# ══════════════════════════════════════════════════════════════════════════════
# GROUP 1 — Extreme Personal Profiles
# ══════════════════════════════════════════════════════════════════════════════
g1 = [
    run_test(
        "Age 13 — youngest user",
        {"age": 13, "age_group": 0, "bmi": 18.0, "bmi_category": 0},
        note="No expectation — observing behaviour"
    ),
    run_test(
        "Age 45, PCOS, BMI 38 — high risk profile",
        {"age": 45, "age_group": 3, "bmi": 38.0, "bmi_category": 3,
         "condition_PCOS": 1, "condition_none": 0},
        expect_spotting=True,
        note="High risk profile should predict spotting"
    ),
    run_test(
        "BMI 14 — severely underweight",
        {"bmi": 14.0, "bmi_category": 0},
        note="Observing — no clear expectation"
    ),
    run_test(
        "BMI 40 — obese",
        {"bmi": 40.0, "bmi_category": 3},
        note="Observing — no clear expectation"
    ),
    run_test(
        "Endometriosis with strong history",
        {"condition_endometriosis": 1, "condition_none": 0,
         "personal_spotting_rate": 0.8,
         "prev_cycle_had_spotting": 1, "consecutive_spotting_months": 4},
        expect_spotting=True,
        note="Strong history added — should predict spotting"
    ),
    run_test(
        "Thyroid condition",
        {"condition_thyroid": 1, "condition_none": 0},
        note="Observing — mild effect expected"
    ),
]

# ══════════════════════════════════════════════════════════════════════════════
# GROUP 2 — Extreme Stress and Lifestyle
# ══════════════════════════════════════════════════════════════════════════════
g2 = [
    run_test(
        "Max stress — Stress 10, Sleep 1, Mood 1",
        {"stress_level": 10, "sleep_quality": 1, "mood_score": 1,
         "stress_trend": 2, "rolling_avg_stress_3m": 10},
        expect_spotting=True,
        note="Extreme stress should push spotting probability above 50%"
    ),
    run_test(
        "Min stress — Stress 1, Sleep 10, Mood 10",
        {"stress_level": 1, "sleep_quality": 10, "mood_score": 10,
         "stress_trend": 0, "rolling_avg_stress_3m": 1,
         "personal_spotting_rate": 0.2,
         "prev_cycle_had_spotting": 0, "consecutive_spotting_months": 0},
        expect_spotting=False,
        note="Low stress + low history should predict no spotting"
    ),
    run_test(
        "High activity + high stress",
        {"activity_level": 2, "stress_level": 9, "stress_trend": 2},
        note="Conflicting signals — observing"
    ),
    run_test(
        "Travel disruption + illness + high stress",
        {"travel_disruption": 1, "recent_illness": 1,
         "stress_level": 8, "stress_trend": 2,
         "personal_spotting_rate": 0.7},
        expect_spotting=True,
        note="Multiple disruptions combined should predict spotting"
    ),
    run_test(
        "Low activity, low hydration",
        {"activity_level": 0, "hydration_level": 1},
        note="Observing — weak signals"
    ),
]

# ══════════════════════════════════════════════════════════════════════════════
# GROUP 3 — History Extremes (FIXED — consistent history)
# ══════════════════════════════════════════════════════════════════════════════
g3 = [
    run_test(
        "12 consecutive spotting months",
        {"consecutive_spotting_months": 12,
         "prev_cycle_had_spotting": 1, "two_months_ago_spotting": 1,
         "personal_spotting_rate": 0.9},
        note="Long streak — model may reduce probability slightly"
    ),
    run_test(
        "First time user — no history",
        {"consecutive_spotting_months": 0,
         "prev_cycle_had_spotting": 0, "two_months_ago_spotting": 0,
         "personal_spotting_rate": 0.5,
         "prev_cycle_gap": 3, "gap_variability": 0.5},
        note="No history — model relies on static features only"
    ),
    run_test(
        "Spotting rate = 0 — consistent no-spotting history",
        {"personal_spotting_rate": 0.0,
         "prev_cycle_had_spotting": 0,
         "two_months_ago_spotting": 0,
         "consecutive_spotting_months": 0,
         "condition_none": 1, "stress_level": 3,
         "contraceptive_pill": 1, "contraceptive_none": 0},
        expect_spotting=False,
        note="FIXED — history now consistent with rate=0"
    ),
    run_test(
        "Spotting rate = 1 — consistent always-spotting history",
        {"personal_spotting_rate": 1.0,
         "prev_cycle_had_spotting": 1,
         "two_months_ago_spotting": 1,
         "consecutive_spotting_months": 6,
         "condition_PCOS": 1, "condition_none": 0,
         "stress_level": 7, "stress_trend": 2},
        expect_spotting=True,
        note="FIXED — history now consistent with rate=1"
    ),
    run_test(
        "Reset month — previous was normal, expect spotting this month",
        {"prev_cycle_had_spotting": 0,
         "consecutive_spotting_months": 0,
         "personal_spotting_rate": 0.7},
        expect_spotting=True,
        note="Normal month followed by spotting is the designed pattern"
    ),
]

# ══════════════════════════════════════════════════════════════════════════════
# GROUP 4 — Gap Extremes (FIXED — using spotting_base so Model 2 runs)
# ══════════════════════════════════════════════════════════════════════════════
g4 = [
    run_test(
        "Base gap = 1, high stress → expect short gap",
        {"personal_base_gap": 1, "prev_cycle_gap": 1,
         "stress_level": 9, "stress_trend": 2},
        base="spotting",
        expect_gap_direction="low",
        note="FIXED — using spotting base so Model 2 runs"
    ),
    run_test(
        "Base gap = 5, low stress → expect long gap",
        {"personal_base_gap": 5, "prev_cycle_gap": 5,
         "stress_level": 2, "stress_trend": 0},
        base="spotting",
        expect_gap_direction="high",
        note="FIXED — using spotting base so Model 2 runs"
    ),
    run_test(
        "Base gap = 3, average inputs → expect mid gap",
        {"personal_base_gap": 3, "prev_cycle_gap": 3},
        base="spotting",
        expect_gap_direction="mid",
        note="Average inputs should give mid-range gap"
    ),
    run_test(
        "Maximum gap variability",
        {"gap_variability": 2.0},
        base="spotting",
        note="Observing gap with high variability"
    ),
    run_test(
        "Zero gap variability — perfectly consistent",
        {"gap_variability": 0.0, "personal_base_gap": 3},
        base="spotting",
        note="Observing gap with zero variability"
    ),
    run_test(
        "Cycle length 21 days — very short",
        {"cycle_length": 21, "cycle_length_deviation": -7},
        base="spotting",
        note="Short cycle — observing gap"
    ),
    run_test(
        "Cycle length 45 days — very long",
        {"cycle_length": 45, "cycle_length_deviation": 17},
        base="spotting",
        note="Long cycle — observing gap"
    ),
]

# ══════════════════════════════════════════════════════════════════════════════
# GROUP 5 — Contradictory Inputs
# ══════════════════════════════════════════════════════════════════════════════
g5 = [
    run_test(
        "Spotting rate=1 but 0 consecutive months",
        {"personal_spotting_rate": 1.0,
         "consecutive_spotting_months": 0,
         "prev_cycle_had_spotting": 0},
        note="Conflicting — rate says always, history says never recently"
    ),
    run_test(
        "Base gap=1 but prev gap=5",
        {"personal_base_gap": 1, "prev_cycle_gap": 5},
        base="spotting",
        note="Which signal dominates — personal history or last month?"
    ),
    run_test(
        "PCOS + pill — condition raises, pill lowers",
        {"condition_PCOS": 1, "condition_none": 0,
         "contraceptive_pill": 1, "contraceptive_none": 0},
        note="Net effect of opposing signals"
    ),
    run_test(
        "High stress + pill contraceptive",
        {"stress_level": 9, "stress_trend": 2,
         "contraceptive_pill": 1, "contraceptive_none": 0},
        note="Stress raises probability, pill lowers it"
    ),
    run_test(
        "Endometriosis + implant contraceptive",
        {"condition_endometriosis": 1, "condition_none": 0,
         "contraceptive_implant": 1, "contraceptive_none": 0},
        note="Condition raises, implant lowers — net effect?"
    ),
    run_test(
        "All risk factors maximum",
        {"condition_PCOS": 1, "condition_none": 0,
         "stress_level": 10, "sleep_quality": 1,
         "travel_disruption": 1, "recent_illness": 1,
         "personal_spotting_rate": 1.0,
         "consecutive_spotting_months": 6,
         "prev_cycle_had_spotting": 1},
        expect_spotting=True,
        note="Everything pointing to spotting — should definitely predict Yes"
    ),
    run_test(
        "All risk factors minimum",
        {"condition_none": 1,
         "contraceptive_pill": 1, "contraceptive_none": 0,
         "stress_level": 1, "sleep_quality": 10,
         "travel_disruption": 0, "recent_illness": 0,
         "personal_spotting_rate": 0.0,
         "consecutive_spotting_months": 0,
         "prev_cycle_had_spotting": 0, "two_months_ago_spotting": 0},
        expect_spotting=False,
        note="Everything pointing away from spotting — should predict No"
    ),
]

# ══════════════════════════════════════════════════════════════════════════════
# COMPILE AND DISPLAY RESULTS
# ══════════════════════════════════════════════════════════════════════════════
all_groups = {
    "Group 1 — Extreme Personal Profiles"  : g1,
    "Group 2 — Extreme Stress and Lifestyle": g2,
    "Group 3 — History Extremes (Fixed)"   : g3,
    "Group 4 — Gap Extremes (Fixed)"       : g4,
    "Group 5 — Contradictory Inputs"       : g5,
}

all_results = []

print("\n" + "="*75)
print("  CYCLEAWARE — EDGE CASE TEST RESULTS v2")
print("="*75)

for group_name, results in all_groups.items():
    print(f"\n{group_name}")
    print("-" * 75)
    df = pd.DataFrame(results)[["Test","Spotting %","Predicted","Gap","Pass","Note"]]
    print(df.to_string(index=False))
    all_results.extend(results)

# Summary
total   = len(all_results)
passed  = sum(1 for r in all_results if r["Pass"] == "PASS")
failed  = sum(1 for r in all_results if r["Pass"] == "FAIL")
errors  = sum(1 for r in all_results if r["Pass"] == "ERROR")
observe = total - passed - failed - errors

print("\n" + "="*75)
print("  SUMMARY")
print("="*75)
print(f"  Total tests        : {total}")
print(f"  Passed             : {passed}")
print(f"  Failed             : {failed}")
print(f"  Errors             : {errors}")
print(f"  Observation only   : {observe}")
print(f"  Pass rate          : {round(passed/(passed+failed)*100) if (passed+failed) > 0 else 0}%")
print("="*75)

if failed > 0:
    print("\n  FAILED TESTS:")
    for r in all_results:
        if r["Pass"] == "FAIL":
            print(f"  - {r['Test']}")
            print(f"    Predicted: {r['Predicted']} | Gap: {r['Gap']} | Note: {r['Note']}")

if errors > 0:
    print("\n  ERRORS (check if backend is running):")
    for r in all_results:
        if r["Pass"] == "ERROR":
            print(f"  - {r['Test']} | {r['Status']}")

print("\n  GAP PREDICTIONS (Model 2 results):")
for r in all_results:
    if r["Gap"] != "N/A" and r["Gap"] != "-":
        print(f"  - {r['Test']} → Gap: {r['Gap']} day(s) | Spotting: {r['Spotting %']}")
