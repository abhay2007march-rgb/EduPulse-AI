"""Train and save the demo model from the reproducible synthetic dataset."""
from pathlib import Path
import pandas as pd

from edupulse_ml import export_metrics, save_bundle, suggest_excluded_columns, train_regression_model
from generate_data import generate_student_data

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "student_performance_demo.csv"


def main() -> None:
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not DATA_FILE.exists():
        generate_student_data().to_csv(DATA_FILE, index=False)
    data = pd.read_csv(DATA_FILE)
    excluded = suggest_excluded_columns(data.columns.tolist(), "final_score")
    bundle = train_regression_model(data, "final_score", excluded)
    save_bundle(bundle)
    export_metrics(bundle)
    print("EduPulse AI demo model trained successfully")
    print(f"Rows: {bundle['sample_count']}; features: {len(bundle['feature_columns'])}")
    print(f"MAE: {bundle['metrics']['mae']:.2f}")
    print(f"RMSE: {bundle['metrics']['rmse']:.2f}")
    print(f"R2: {bundle['metrics']['r2']:.3f}")


if __name__ == "__main__":
    main()
