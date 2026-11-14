import unittest
import pandas as pd
from tools import profile_dataset, detect_quality_issues, detect_anomalies, create_cleaning_preview

class ToolTests(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            "score": [10, 11, 12, 13, 100],
            "name": ["A", "B", "B", None, "E"],
            "constant": [1, 1, 1, 1, 1]
        })

    def test_profile_dimensions(self):
        result = profile_dataset(self.df)
        self.assertEqual(result["rows"], 5)
        self.assertEqual(result["columns"], 3)

    def test_quality_issues(self):
        result = detect_quality_issues(self.df)
        self.assertEqual(result["total_missing_cells"], 1)
        self.assertEqual(result["constant_columns"], ["constant"])

    def test_anomaly_tool(self):
        result = detect_anomalies(self.df)
        self.assertIn("score", result["columns"])
        self.assertGreaterEqual(result["columns"]["score"]["candidate_count"], 1)

    def test_preview_does_not_modify_source(self):
        before = self.df.copy(deep=True)
        result = create_cleaning_preview(self.df)
        pd.testing.assert_frame_equal(self.df, before)
        self.assertFalse(result["source_modified"])

if __name__ == "__main__":
    unittest.main()
