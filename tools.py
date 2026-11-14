
"""Deterministic, auditable data-analysis tools used by the agent."""

from __future__ import annotations

import pandas as pd
import numpy as np


def _json_safe(value):
    """Convert values into JSON-compatible Python values."""
    if pd.isna(value):
        return None

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        return float(value)

    return str(value)


def profile_dataset(df: pd.DataFrame) -> dict:
    """Generate a basic dataset profile, including numeric summaries."""

    numeric = df.select_dtypes(include="number")

    # Handle datasets with no numeric columns
    if numeric.empty:
        numeric_summary = {}
    else:
        numeric_summary = numeric.describe().round(3).to_dict()

    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": [str(c) for c in df.columns],
        "data_types": {
            str(k): str(v) for k, v in df.dtypes.items()
        },
        "numeric_summary": numeric_summary,
        "memory_usage_kb": round(
            float(df.memory_usage(deep=True).sum() / 1024), 2
        ),
    }


def detect_quality_issues(df: pd.DataFrame) -> dict:
    """Detect missing values, duplicates, constant columns and blank text."""

    missing = df.isna().sum()
    duplicate_rows = int(df.duplicated().sum())

    constant = [
        str(c)
        for c in df.columns
        if df[c].nunique(dropna=False) <= 1
    ]

    missing_report = {
        str(col): {
            "count": int(count),
            "percentage": round(
                float(count / max(len(df), 1) * 100), 2
            )
        }
        for col, count in missing.items()
        if count > 0
    }

    suspicious_text = {}

    for col in df.select_dtypes(include=["object", "string"]).columns:
        values = df[col].dropna().astype(str)

        whitespace = int(values.str.strip().eq("").sum())

        if whitespace:
            suspicious_text[str(col)] = {
                "blank_strings": whitespace
            }

    return {
        "missing_values": missing_report,
        "duplicate_rows": duplicate_rows,
        "constant_columns": constant,
        "blank_text_values": suspicious_text,
        "total_missing_cells": int(missing.sum()),
    }


def detect_anomalies(df: pd.DataFrame) -> dict:
    """
    Detect potential numeric outliers using the IQR method.

    Outliers are screening candidates, not confirmed errors.
    """

    report = {}

    for col in df.select_dtypes(include="number").columns:

        series = pd.to_numeric(
            df[col], errors="coerce"
        ).dropna()

        # Skip columns with fewer than four numeric values
        if len(series) < 4:
            continue

        q1, q3 = series.quantile([0.25, 0.75])

        iqr = q3 - q1

        # Skip columns with no variation
        if iqr == 0:
            continue

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        mask = (series < lower) | (series > upper)

        report[str(col)] = {
            "lower_bound": round(float(lower), 4),
            "upper_bound": round(float(upper), 4),
            "candidate_count": int(mask.sum()),
            "candidate_percentage": round(
                float(mask.mean() * 100), 2
            ),
            "sample_values": [
                float(x) for x in series[mask].head(5).tolist()
            ],
        }

    return {
        "method": "1.5 x IQR rule",
        "interpretation": "Candidates require domain review.",
        "columns": report,
    }


def create_cleaning_preview(df: pd.DataFrame) -> dict:
    """
    Suggest possible cleaning actions without modifying the source dataset.
    """

    actions = []

    for col in df.columns:

        missing_count = int(df[col].isna().sum())

        if missing_count:

            if pd.api.types.is_numeric_dtype(df[col]):
                strategy = (
                    "Review median imputation; compare with "
                    "domain rules first."
                )
            else:
                strategy = (
                    "Review mode or explicit 'Unknown' category; "
                    "do not assume."
                )

            actions.append({
                "column": str(col),
                "issue": "missing values",
                "count": missing_count,
                "suggestion": strategy,
            })

        if df[col].dtype == "object":

            whitespace = int(
                df[col].dropna().astype(str).str.strip().eq("").sum()
            )

            if whitespace:
                actions.append({
                    "column": str(col),
                    "issue": "blank strings",
                    "count": whitespace,
                    "suggestion": (
                        "Convert whitespace-only strings to missing "
                        "values after review."
                    ),
                })

    duplicates = int(df.duplicated().sum())

    if duplicates:
        actions.append({
            "column": "ALL",
            "issue": "duplicate rows",
            "count": duplicates,
            "suggestion": (
                "Inspect duplicate records before removal; confirm "
                "whether repeated events are valid."
            ),
        })

    return {
        "source_modified": False,
        "suggested_actions": actions,
        "message": (
            "This is a review plan only. The original dataset "
            "has not been modified."
        ),
    }
