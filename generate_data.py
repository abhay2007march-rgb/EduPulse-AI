"""Create a reproducible, clearly-labelled synthetic student dataset."""
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
DATA_PATH = BASE / "data" / "student_performance_demo.csv"


def generate_student_data(n: int = 1500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    departments = np.array(["CSE", "ECE", "EEE", "ME", "CE", "IT"])
    internet = np.array(["Good", "Average", "Limited"])
    activity = np.array(["Sports", "Arts", "Technical Club", "Volunteering", "None"])
    attendance = np.clip(rng.normal(82, 11, n), 42, 100)
    study_hours = np.clip(rng.gamma(3.0, 3.0, n) + 1.5, 1, 30)
    assignments = np.clip(0.48 * attendance + rng.normal(31, 15, n), 15, 100)
    previous_score = np.clip(rng.normal(67, 14, n), 18, 98)
    internal_score = np.clip(0.32 * previous_score + 0.36 * study_hours * 2.3 + 0.20 * assignments + rng.normal(13, 10, n), 12, 100)
    sleep_hours = np.clip(rng.normal(7.1, 1.1, n), 3.5, 10)
    participation = np.clip(rng.beta(2.3, 2.0, n) * 100, 0, 100)
    commute = np.clip(rng.gamma(2.1, 16, n), 2, 120)
    semester = rng.integers(1, 9, n)
    dept = rng.choice(departments, n, p=[0.30, 0.18, 0.13, 0.14, 0.10, 0.15])
    internet_access = rng.choice(internet, n, p=[0.55, 0.34, 0.11])
    extracurricular = rng.choice(activity, n, p=[0.20, 0.12, 0.25, 0.10, 0.33])
    study_component = np.clip(study_hours / 30 * 100, 0, 100)
    commute_penalty = np.maximum(commute - 45, 0) * 0.06
    sleep_component = 100 - np.abs(sleep_hours - 8.0) * 13
    internet_bonus = np.select([internet_access == "Good", internet_access == "Average"], [2.2, 0.5], default=-2.2)
    activity_bonus = np.select([extracurricular == "Technical Club", extracurricular == "Sports"], [2.0, 1.0], default=0.0)
    final_score = (
        0.23 * previous_score + 0.24 * internal_score + 0.16 * attendance
        + 0.14 * assignments + 0.10 * study_component + 0.06 * participation
        + 0.05 * sleep_component + internet_bonus + activity_bonus
        - commute_penalty + rng.normal(0, 5.8, n)
    )
    final_score = np.clip(final_score, 0, 100).round(1)
    return pd.DataFrame({
        "student_id": [f"STU{idx:05d}" for idx in range(1, n + 1)],
        "department": dept, "semester": semester,
        "attendance_rate": attendance.round(1),
        "study_hours_per_week": study_hours.round(1),
        "assignment_completion": assignments.round(1),
        "previous_score": previous_score.round(1),
        "internal_score": internal_score.round(1),
        "sleep_hours": sleep_hours.round(1),
        "class_participation": participation.round(1),
        "internet_access": internet_access,
        "extracurricular_activity": extracurricular,
        "commute_minutes": commute.round(0).astype(int),
        "final_score": final_score,
    })


if __name__ == "__main__":
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame = generate_student_data()
    frame.to_csv(DATA_PATH, index=False)
    print(f"Created {len(frame):,} synthetic records at {DATA_PATH}")
