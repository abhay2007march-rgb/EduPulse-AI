import unittest
import pandas as pd

from edupulse_ml import make_predictions, suggest_excluded_columns, train_regression_model
from generate_data import generate_student_data


class EduPulseModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = generate_student_data(180, seed=7)
        excluded = suggest_excluded_columns(cls.data.columns.tolist(), "final_score")
        cls.bundle = train_regression_model(cls.data, "final_score", excluded)

    def test_predictions_include_score_and_risk(self):
        features = self.data.drop(columns=["final_score"]).head(8)
        result = make_predictions(self.bundle, features)
        self.assertEqual(len(result), 8)
        self.assertIn("predicted_final_score", result.columns)
        self.assertIn("predicted_risk", result.columns)

    def test_missing_feature_is_reported(self):
        with self.assertRaisesRegex(ValueError, "missing required feature"):
            make_predictions(self.bundle, pd.DataFrame({"unrelated": [1]}))

    def test_training_metrics_are_available(self):
        self.assertGreater(self.bundle["metrics"]["test_rows"], 0)
        self.assertEqual(self.bundle["target_column"], "final_score")


if __name__ == "__main__":
    unittest.main()
