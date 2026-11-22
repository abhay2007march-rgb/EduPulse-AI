# EduPulse AI — Advanced Student Analytics

EduPulse AI is a Streamlit dashboard for exploring student data, training a Random Forest model, estimating academic outcomes, and identifying students who may benefit from early support.

> **Responsible use:** The built-in demonstration data is synthetic. Demonstration metrics are not evidence of real-world predictive accuracy. Validate with representative, consented institutional data before using predictions in practice. Predictions must never be the sole basis for grading, admission, discipline, scholarships, or other high-impact decisions.

## Features

- Executive overview with score distributions, pass-rate estimates, attendance summaries, and department comparisons.
- Data explorer with previews, missing-value checks, descriptive statistics, and correlation analysis.
- CSV and Excel uploads for analysis and model retraining.
- Random Forest regression pipeline with numeric/categorical preprocessing, missing-value imputation, and one-hot encoding.
- Held-out validation metrics: MAE, RMSE, and R².
- Actual-vs-predicted and residual visualizations, plus feature-importance analysis.
- Individual student predictions with estimated risk bands.
- Batch predictions with downloadable CSV output.
- Export of the active trained model as a Joblib file.
- Render Blueprint configuration included.

## Project layout

```text
EduPulse-AI/
├── app.py
├── edupulse_ml.py
├── generate_data.py
├── train_model.py
├── requirements.txt
├── runtime.txt
├── render.yaml
├── start.bat
├── retrain_model.bat
└── tests/
    └── test_ml.py
```

The demo dataset and model are generated locally on first launch when they are not present. Generated files are ignored by Git so large binary artifacts and generated student rows are not committed.

## Run on Windows (Python 3.12)

1. Install Python 3.12 and Git.
2. Download or clone this repository and open its folder in VS Code.
3. Run `start.bat`, or use the manual commands below.

### Manual setup

```cmd
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local Streamlit URL shown in the terminal (usually http://localhost:8501).

On first startup, EduPulse AI creates 1,500 synthetic demonstration records and trains a Random Forest model if no saved model exists. The generated model is stored under `models/` on the local machine.

## Retrain the demonstration model

Run `retrain_model.bat`, or:

```cmd
python generate_data.py
python train_model.py
```

The training script saves `models/demo_model.joblib` and `artifacts/model_metrics.json`.

## Train with your own dataset

1. Prepare a CSV or Excel file with at least 30 rows and a numeric outcome column (for example, `final_score`).
2. Open EduPulse AI and upload it in the sidebar.
3. Select **Model Training & Evaluation**.
4. Choose the target column and review automatically suggested exclusions.
5. Exclude identifiers and any fields that leak the target or would not be known at prediction time.
6. Click **Train / Retrain model** and inspect the held-out metrics and diagnostic plots.
7. Use Individual Prediction or Batch Predictions with the active feature schema.

Numeric missing values are median-imputed. Categorical missing values are most-frequent-imputed, and unseen categories are handled by the encoder. At least one usable feature and two distinct target values are required.

## Run tests

After installing requirements:

```cmd
python -m unittest discover -s tests -v
```

## Deploy to Render

This repository includes `render.yaml`.

1. Push the project to GitHub.
2. In Render, choose **New → Blueprint**.
3. Connect `abhay2007march-rgb/EduPulse-AI`.
4. Render reads `render.yaml` and creates the Streamlit web service.
5. Wait for the deployment to finish and open the service URL.

The service builds with `pip install -r requirements.txt` and starts with Streamlit bound to Render's `$PORT`.

## Model limitations

The demo labels are generated from a simulated scoring formula. A model trained on this synthetic data is only a demonstration of the ML workflow. Before institutional use, evaluate temporal and student-level data leakage, subgroup performance, calibration, drift, privacy, consent, and local policy. Involve educators in interpreting predictions and providing support.

## License

Educational prototype. Add an institutional or open-source license before redistribution.
