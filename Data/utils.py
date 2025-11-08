import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, Optional, List, Tuple
from scipy.stats import chi2_contingency
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF
from sklearn.model_selection import cross_val_score
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


def tune_model(
    model_type: str,
    X: pd.DataFrame,
    y: pd.Series,
    preprocessor: ColumnTransformer,
    n_trials: int = 50,
) -> Dict[str, object]:
    """
    Tune hyperparameters for a specified classification model using Optuna.

    Parameters:
        model_type (str): The name of the model to tune.
                          Supported: 'Logistic Regression', 'Gradient Boosting', 'XGBoost', 'LightGBM', 'CatBoost'.
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target variable.
        preprocessor (ColumnTransformer): Preprocessing pipeline for numeric and categorical features.
        n_trials (int, optional): Number of Optuna trials to run. Default is 50.

    Returns:
        Dict[str, object]: Dictionary of best hyperparameters found by Optuna.
    """

    def objective(trial):
        if model_type == "Gradient Boosting":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.3, log=True
                ),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            }
            model = GradientBoostingClassifier(**params, random_state=42)

        elif model_type == "XGBoost":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.3, log=True
                ),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "gamma": trial.suggest_float("gamma", 0, 5),
                "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1, 15),
            }
            model = XGBClassifier(
                **params,
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=42,
            )

        elif model_type == "LightGBM":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.3, log=True
                ),
                "num_leaves": trial.suggest_int("num_leaves", 20, 100),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0, 10),
                "reg_lambda": trial.suggest_float("reg_lambda", 0, 10),
                "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1, 15),
            }
            model = LGBMClassifier(**params, random_state=42, verbose=-1)

        elif model_type == "CatBoost":
            params = {
                "iterations": trial.suggest_int("iterations", 200, 800),
                "depth": trial.suggest_int("depth", 4, 10),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.3, log=True
                ),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
                "border_count": trial.suggest_int("border_count", 32, 255),
                "auto_class_weights": "Balanced",
            }
            model = CatBoostClassifier(**params, verbose=0, random_state=42)

        elif model_type == "Logistic Regression":
            params = {
                "C": trial.suggest_float("C", 0.01, 10, log=True),
                "penalty": trial.suggest_categorical("penalty", ["l2"]),
                "solver": trial.suggest_categorical("solver", ["liblinear", "lbfgs"]),
                "class_weight": "balanced",
                "max_iter": 1000,
            }
            model = LogisticRegression(**params, random_state=42)

        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        pipe = Pipeline(
            [
                ("preprocessor", preprocessor),
                ("classifier", model),
            ]
        )

        y_scores = cross_val_predict(
            pipe,
            X,
            y,
            cv=StratifiedKFold(5, shuffle=True, random_state=42),
            method="predict_proba",
            n_jobs=1,
        )[:, 1]

        return average_precision_score(y, y_scores)

    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, n_jobs=1)

    return study.best_params

    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, n_jobs=1)

    return study.best_params


def evaluate_feature_addition_auc(
    X: pd.DataFrame,
    y: pd.Series,
    base_features: List[str],
    added_feature: str,
    model=None,
    cv: int = 5,
    scoring: str = "roc_auc",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Compare AUC performance with and without a specific feature using LightGBM.

    Parameters:
    - X: Full feature DataFrame
    - y: Target variable
    - base_features: List of features for baseline model
    - added_feature: Feature to test for added value
    - model: Model to use (default: LightGBM)
    - cv: Cross-validation folds
    - scoring: Metric to use (default: roc_auc)
    - random_state: Reproducibility

    Returns:
    - DataFrame with AUC scores for base and extended models
    """
    N_pos = np.sum(y == 1)
    N_neg = np.sum(y == 0)
    scale_pos_weight_val = N_neg / N_pos

    if model is None:
        model = LGBMClassifier(
            random_state=random_state, verbose=-1, scale_pos_weight=scale_pos_weight_val
        )

    X_base = X[base_features]
    base_auc = cross_val_score(model, X_base, y, cv=cv, scoring=scoring).mean()

    extended_features = base_features + [added_feature]
    X_ext = X[extended_features]
    ext_auc = cross_val_score(model, X_ext, y, cv=cv, scoring=scoring).mean()

    result = pd.DataFrame(
        {
            "Model": ["Base", f"Base + {added_feature}"],
            "ROC-AUC": [base_auc, ext_auc],
        }
    )

    return result.round(4)


def add_outlier_feature(
    df, column, new_col=None, method="upper", multiplier=1.5
) -> pd.DataFrame:
    """
    Adds a binary outlier feature to the DataFrame based on IQR.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - column (str): Name of the numeric column to analyze.
    - new_col (str): Name for the new outlier column. If None, defaults to `{column}_outlier`.
    - method (str): 'upper', 'lower', or 'both' to detect outliers.
    - multiplier (float): IQR multiplier to define the threshold.

    Returns:
    - pd.DataFrame: DataFrame with the new outlier column added.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR

    if method == "upper":
        outlier_mask = df[column] > upper_bound
    elif method == "lower":
        outlier_mask = df[column] < lower_bound
    elif method == "both":
        outlier_mask = (df[column] > upper_bound) | (df[column] < lower_bound)
    else:
        raise ValueError("method must be 'upper', 'lower', or 'both'")

    if new_col is None:
        new_col = f"{column}_outlier"

    df[new_col] = outlier_mask.astype(int)
    return df


def calculate_vif(dataframe: pd.DataFrame, target_column: str = None) -> pd.DataFrame:
    """
    Calculate Variance Inflation Factor (VIF) for numeric features in a DataFrame.

    Parameters:
        dataframe (pd.DataFrame): The input DataFrame.
        target_column (str): Optional. Column to exclude (e.g., target like 'stroke').

    Returns:
        pd.DataFrame: VIF scores sorted descending by VIF.
    """

    if target_column and target_column in dataframe.columns:
        X = dataframe.drop(columns=target_column)
    else:
        X = dataframe.copy()

    X = X.select_dtypes(include=[np.number])

    X = sm.add_constant(X)

    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [VIF(X.values, i) for i in range(X.shape[1])]
    vif_data["VIF"] = vif_data["VIF"].round(2)

    return vif_data.sort_values(by="VIF", ascending=False).reset_index(drop=True)


def bootstrap_median_diff(
    data1, data2, num_iterations=1000, ci=95
) -> Tuple[float, float]:
    """
    Computes the bootstrap confidence interval for the difference in medians
    between two datasets.

    This function randomly resamples the two input datasets with replacement
    to simulate the sampling distribution of the difference in medians. It
    calculates the confidence interval for the difference based on the
    bootstrap resamples.

    Parameters:
    -----------
    data1 : array-like
        First dataset.
    data2 : array-like
        Second dataset.
    num_iterations : int, optional (default=1000)
        Number of bootstrap iterations to perform.
    ci : float, optional (default=95)
        The confidence level for the interval (e.g., 95 for a 95% confidence interval).

    Returns:
    --------
    lower : float
        The lower bound of the bootstrap confidence interval.
    upper : float
        The upper bound of the bootstrap confidence interval.
    """
    boot_diffs = []
    n1 = len(data1)
    n2 = len(data2)
    for i in range(num_iterations):
        boot_sample1 = np.random.choice(data1, size=n1, replace=True)
        boot_sample2 = np.random.choice(data2, size=n2, replace=True)
        boot_diffs.append(np.median(boot_sample1) - np.median(boot_sample2))
    lower = np.percentile(boot_diffs, (100 - ci) / 2)
    upper = np.percentile(boot_diffs, 100 - (100 - ci) / 2)
    return lower, upper


def stacked_bar_with_percent(
    data: pd.DataFrame,
    column_x: str,
    column_y: str = "stroke",
    figsize: Tuple[int, int] = (8, 4),
) -> None:
    """
    Plot stacked bar chart with bars sized by actual frequency and
    annotated with percentages for binary target analysis.
    """
    # Get counts
    count_table = pd.crosstab(data[column_x], data[column_y])

    # Normalize by all data (not index)
    percent_table = count_table.div(count_table.sum(axis=1), axis=0) * 100

    ax = count_table.plot(
        kind="bar",
        stacked=True,
        figsize=figsize,
    )

    # Annotate with percentage labels
    for i, category in enumerate(count_table.index):
        bottom = 0
        for stroke_value in count_table.columns:
            count = count_table.loc[category, stroke_value]
            percent = percent_table.loc[category, stroke_value]
            if count > 0:
                ax.text(
                    i,
                    bottom + count / 2,
                    f"{percent:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="black",
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=2),
                )
            bottom += count

    plt.title(f"Stroke prevalence by {column_x}", pad=15)
    plt.xlabel(column_x)
    plt.ylabel("Number of patients")
    plt.legend(
        title="Stroke",
        labels=["No", "Yes"],
        loc="upper right",
        frameon=True,
    )
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()


def crosstab_chi2_test(df: pd.DataFrame, col_x: str, col_y: str) -> None:
    """
    Computes and prints a normalized crosstab and performs Chi-squared test of independence.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data
        col_x (str): The name of the row variable (e.g., 'FrequentFlyer')
        col_y (str): The name of the column variable (e.g., 'TravelInsurance')
    """
    print(f"\n Crosstab of {col_x} vs. {col_y}\n")

    cross_tab_norm = pd.crosstab(df[col_x], df[col_y], normalize="index").round(2) * 100
    print("Normalized Crosstab (%):")
    print(cross_tab_norm)

    cross_tab_counts = pd.crosstab(df[col_x], df[col_y])
    chi2, p, dof, expected = chi2_contingency(cross_tab_counts)

    print("\nChi-squared test results:")
    print(f"Chi-squared Statistic: {chi2:.4f}")
    print(f"Degrees of Freedom: {dof}")
    print(f"P-value: {p:.4f}")

    if p < 0.05:
        print(" Statistically significant association (p < 0.05)")
    else:
        print(" No significant association (p ≥ 0.05)")


def plot_distribution_numerical(
    data: pd.DataFrame,
    column: str,
    target: str = "stroke",
    figsize: Tuple[int, int] = (8, 6),
    bins: int = 10,
) -> None:
    """
    Plot comparative analysis of a numerical feature vs target variable
    including general and conditional distributions and boxplots.

    Parameters:
    - data: Input DataFrame
    - column: Numerical column to analyze
    - target: Target variable name (default: stroke)
    - figsize: Figure size (width, height)
    - bins: Number of histogram bins
    """
    plt.figure(figsize=figsize)

    plt.subplot(2, 2, 1)
    sns.boxplot(
        y=data[column],
        showmeans=True,
        meanprops={
            "marker": "o",
            "markerfacecolor": "white",
            "markeredgecolor": "black",
        },
    )
    plt.title(f"General Boxplot of {column}")
    plt.xlabel("")
    plt.ylabel(column)

    plt.subplot(2, 2, 2)
    sns.boxplot(
        x=target,
        y=column,
        data=data,
        showmeans=True,
        meanprops={
            "marker": "o",
            "markerfacecolor": "white",
            "markeredgecolor": "black",
        },
    )
    plt.title(f"{column.capitalize()} by {target}")
    plt.xlabel(target)
    plt.ylabel(column)

    plt.subplot(2, 2, 3)
    sns.histplot(data[column].dropna(), bins=bins, kde=True, color="skyblue")
    plt.title(f"Overall distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")

    plt.subplot(2, 2, 4)
    sns.histplot(
        data=data,
        x=column,
        hue=target,
        bins=bins,
        kde=True,
        element="step",
        common_norm=False,
    )
    plt.title(f"{column.capitalize()} distribution by {target}")
    plt.xlabel(column)
    plt.ylabel("Density")

    plt.tight_layout()
    plt.show()
