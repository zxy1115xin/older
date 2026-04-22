from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ELDERLY_THRESHOLD = 60.0


def load_subject_feature_table() -> pd.DataFrame:
    path = Path("output") / "gait_feature_deep_analysis.xlsx"
    if not path.exists():
        raise FileNotFoundError(
            f"未找到 {path}，请先运行 analyze_gait_features.py 生成受试者特征表。"
        )
    return pd.read_excel(path, sheet_name="subject_features")


def build_wide_feature_table(subject_feature_df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [
        "PeakValue",
        "PeakCycle",
        "TroughValue",
        "TroughCycle",
        "ROM",
        "MeanValue",
        "StdValue",
        "AUC",
    ]

    wide_df = (
        subject_feature_df.set_index(["Subject", "Age", "AgeGroup", "SheetName"])[feature_cols]
        .unstack("SheetName")
        .reset_index()
    )
    wide_df.columns = [
        col if isinstance(col, str) else (col[0] if col[1] == "" else f"{col[1]}__{col[0]}")
        for col in wide_df.columns.to_flat_index()
    ]
    wide_df["ElderlyLabel"] = (wide_df["Age"] >= ELDERLY_THRESHOLD).astype(int)
    return wide_df


def get_feature_columns(wide_df: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric_features = [
        col
        for col in wide_df.columns
        if "__" in col and pd.api.types.is_numeric_dtype(wide_df[col])
    ]
    categorical_features = ["AgeGroup"]
    return numeric_features, categorical_features


def build_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    numeric_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric_pipe, numeric_features),
            ("cat", categorical_pipe, categorical_features),
        ]
    )


def evaluate_age_regression(
    wide_df: pd.DataFrame, numeric_features: list[str], categorical_features: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    x = wide_df[numeric_features + categorical_features]
    y = wide_df["Age"]

    ridge_model = Pipeline(
        [
            ("preprocessor", build_preprocessor(numeric_features, categorical_features)),
            ("model", Ridge(alpha=1.0)),
        ]
    )
    rf_model = Pipeline(
        [
            ("preprocessor", build_preprocessor(numeric_features, categorical_features)),
            ("model", RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=1)),
        ]
    )

    ridge_pred = cross_val_predict(ridge_model, x, y, cv=5)
    rf_pred = cross_val_predict(rf_model, x, y, cv=5)

    regression_summary = pd.DataFrame(
        [
            {
                "Model": "Ridge",
                "MAE": float(mean_absolute_error(y, ridge_pred)),
                "R2": float(r2_score(y, ridge_pred)),
            },
            {
                "Model": "RandomForestRegressor",
                "MAE": float(mean_absolute_error(y, rf_pred)),
                "R2": float(r2_score(y, rf_pred)),
            },
        ]
    )

    pred_df = pd.DataFrame(
        {
            "Subject": wide_df["Subject"],
            "ActualAge": y,
            "PredAge_Ridge": ridge_pred,
            "PredAge_RF": rf_pred,
        }
    )
    return regression_summary, pred_df


def evaluate_elderly_classification(
    wide_df: pd.DataFrame, numeric_features: list[str], categorical_features: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    x = wide_df[numeric_features + categorical_features]
    y = wide_df["ElderlyLabel"]

    logit_model = Pipeline(
        [
            ("preprocessor", build_preprocessor(numeric_features, categorical_features)),
            ("model", LogisticRegression(max_iter=4000, class_weight="balanced")),
        ]
    )
    rf_model = Pipeline(
        [
            ("preprocessor", build_preprocessor(numeric_features, categorical_features)),
            ("model", RandomForestClassifier(n_estimators=500, random_state=42, class_weight="balanced", n_jobs=1)),
        ]
    )

    logit_prob = cross_val_predict(logit_model, x, y, cv=5, method="predict_proba")[:, 1]
    rf_prob = cross_val_predict(rf_model, x, y, cv=5, method="predict_proba")[:, 1]
    logit_pred = (logit_prob >= 0.5).astype(int)
    rf_pred = (rf_prob >= 0.5).astype(int)

    classification_summary = pd.DataFrame(
        [
            {
                "Model": "LogisticRegression",
                "Accuracy": float(accuracy_score(y, logit_pred)),
                "ROC_AUC": float(roc_auc_score(y, logit_prob)),
            },
            {
                "Model": "RandomForestClassifier",
                "Accuracy": float(accuracy_score(y, rf_pred)),
                "ROC_AUC": float(roc_auc_score(y, rf_prob)),
            },
        ]
    )

    pred_df = pd.DataFrame(
        {
            "Subject": wide_df["Subject"],
            "ActualAge": wide_df["Age"],
            "ActualElderlyLabel": y,
            "PredProb_Logistic": logit_prob,
            "PredProb_RF": rf_prob,
            "PredLabel_Logistic": logit_pred,
            "PredLabel_RF": rf_pred,
        }
    )
    return classification_summary, pred_df


def extract_feature_importance(
    wide_df: pd.DataFrame, numeric_features: list[str], categorical_features: list[str]
) -> pd.DataFrame:
    x = wide_df[numeric_features + categorical_features]
    y = wide_df["ElderlyLabel"]

    pipeline = Pipeline(
        [
            ("preprocessor", build_preprocessor(numeric_features, categorical_features)),
            ("model", RandomForestClassifier(n_estimators=500, random_state=42, class_weight="balanced", n_jobs=1)),
        ]
    )
    pipeline.fit(x, y)

    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    importance = pipeline.named_steps["model"].feature_importances_
    importance_df = pd.DataFrame(
        {"FeatureName": feature_names, "Importance": importance}
    ).sort_values("Importance", ascending=False)

    importance_df["FeatureName"] = importance_df["FeatureName"].str.replace("num__", "", regex=False)
    importance_df["FeatureName"] = importance_df["FeatureName"].str.replace("cat__", "", regex=False)
    return importance_df.reset_index(drop=True)


def save_reports(
    wide_df: pd.DataFrame,
    regression_summary: pd.DataFrame,
    age_pred_df: pd.DataFrame,
    classification_summary: pd.DataFrame,
    cls_pred_df: pd.DataFrame,
    importance_df: pd.DataFrame,
) -> Path:
    output_path = Path("output") / "age_prediction_model_results.xlsx"
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        wide_df.to_excel(writer, sheet_name="wide_features", index=False)
        regression_summary.to_excel(writer, sheet_name="age_regression_summary", index=False)
        age_pred_df.to_excel(writer, sheet_name="age_regression_preds", index=False)
        classification_summary.to_excel(writer, sheet_name="elderly_cls_summary", index=False)
        cls_pred_df.to_excel(writer, sheet_name="elderly_cls_preds", index=False)
        importance_df.to_excel(writer, sheet_name="feature_importance", index=False)
    return output_path


def main() -> None:
    subject_feature_df = load_subject_feature_table()
    wide_df = build_wide_feature_table(subject_feature_df)
    numeric_features, categorical_features = get_feature_columns(wide_df)

    regression_summary, age_pred_df = evaluate_age_regression(
        wide_df, numeric_features, categorical_features
    )
    classification_summary, cls_pred_df = evaluate_elderly_classification(
        wide_df, numeric_features, categorical_features
    )
    importance_df = extract_feature_importance(wide_df, numeric_features, categorical_features)
    output_path = save_reports(
        wide_df=wide_df,
        regression_summary=regression_summary,
        age_pred_df=age_pred_df,
        classification_summary=classification_summary,
        cls_pred_df=cls_pred_df,
        importance_df=importance_df,
    )

    print(f"已输出年龄预测与老年判别结果: {output_path}")
    print("年龄回归结果:")
    print(regression_summary.to_string(index=False))
    print("老年判别结果:")
    print(classification_summary.to_string(index=False))
    print("最重要的前15个特征:")
    print(importance_df.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
