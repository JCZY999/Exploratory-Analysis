"""Comprehensive, reproducible exploratory data analysis case study."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from generate_data import generate_dataset

ROOT = Path(__file__).parent
OUTPUT = ROOT / "outputs"
NUMERIC = ["sessions", "engagement_score", "pages_viewed", "media_spend", "age", "household_income", "revenue"]
CATEGORICAL = ["channel", "region", "device"]


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    table = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(table)[0]
    n = table.values.sum()
    phi2 = chi2 / n
    r, k = table.shape
    return float(np.sqrt(max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1)) / max(1e-9, min(k - 1, r - 1))))


def robust_outlier_flags(series: pd.Series) -> pd.DataFrame:
    clean = series.dropna()
    q1, q3 = clean.quantile([.25, .75])
    iqr = q3 - q1
    median = clean.median()
    mad = np.median(np.abs(clean - median))
    robust_z = .6745 * (series - median) / (mad if mad else 1)
    return pd.DataFrame({"iqr_outlier": (series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr), "robust_z_outlier": robust_z.abs() > 3.5})


def structural_audit(df: pd.DataFrame) -> dict:
    return {
        "rows": len(df), "columns": len(df.columns), "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_customer_ids": int(df["customer_id"].duplicated().sum()),
        "date_min": str(df["event_date"].min().date()), "date_max": str(df["event_date"].max().date()),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1_048_576, 3)
    }


def profile_columns(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in df.columns:
        series = df[column]
        rows.append({"column": column, "dtype": str(series.dtype), "non_null": int(series.notna().sum()),
                     "missing_pct": float(series.isna().mean()), "unique": int(series.nunique(dropna=True)),
                     "sample": str(series.dropna().iloc[0]) if series.notna().any() else None})
    return pd.DataFrame(rows)


def numeric_profile(df: pd.DataFrame) -> pd.DataFrame:
    result = df[NUMERIC].describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]).T
    result["missing_pct"] = df[NUMERIC].isna().mean()
    result["skew"] = df[NUMERIC].skew()
    result["kurtosis"] = df[NUMERIC].kurtosis()
    for col in NUMERIC:
        flags = robust_outlier_flags(df[col])
        result.loc[col, "iqr_outliers"] = flags["iqr_outlier"].sum()
        result.loc[col, "robust_z_outliers"] = flags["robust_z_outlier"].sum()
    return result


def statistical_tests(df: pd.DataFrame) -> pd.DataFrame:
    clean = df.dropna(subset=["channel", "device", "engagement_score"])
    converted, not_converted = clean.loc[clean.converted == 1, "engagement_score"], clean.loc[clean.converted == 0, "engagement_score"]
    welch = stats.ttest_ind(converted, not_converted, equal_var=False)
    mann = stats.mannwhitneyu(converted, not_converted, alternative="two-sided")
    chi2 = stats.chi2_contingency(pd.crosstab(clean.channel, clean.converted))
    groups = [group.engagement_score.dropna() for _, group in clean.groupby("channel")]
    kruskal = stats.kruskal(*groups)
    return pd.DataFrame([
        {"test":"Welch t-test", "question":"Does engagement differ by conversion?", "statistic":welch.statistic, "p_value":welch.pvalue},
        {"test":"Mann–Whitney U", "question":"Do engagement ranks differ by conversion?", "statistic":mann.statistic, "p_value":mann.pvalue},
        {"test":"Chi-square", "question":"Are channel and conversion associated?", "statistic":chi2.statistic, "p_value":chi2.pvalue},
        {"test":"Kruskal–Wallis", "question":"Does engagement differ across channels?", "statistic":kruskal.statistic, "p_value":kruskal.pvalue}
    ])


def multivariate_analysis(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = ["sessions", "engagement_score", "pages_viewed", "media_spend", "age", "household_income", "channel", "device"]
    work = df[features].copy()
    for col in ["engagement_score", "household_income"]:
        work[col] = work[col].fillna(work[col].median())
    work["device"] = work["device"].fillna("Missing")
    transformer = ColumnTransformer([("num", StandardScaler(), features[:6]), ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), features[6:])])
    matrix = transformer.fit_transform(work)
    pca = PCA(n_components=3, random_state=42).fit_transform(matrix)
    clusters = KMeans(n_clusters=4, random_state=42, n_init=20).fit_predict(matrix)
    scored = df[["customer_id", "converted", "revenue"]].copy()
    scored[["pc1", "pc2", "pc3"]] = pca
    scored["cluster"] = clusters
    summary = scored.groupby("cluster", as_index=False).agg(customers=("customer_id", "count"), conversion_rate=("converted", "mean"), average_revenue=("revenue", "mean"), pc1=("pc1", "mean"), pc2=("pc2", "mean"))
    return scored, summary


def run_eda(df: pd.DataFrame | None = None) -> dict:
    OUTPUT.mkdir(exist_ok=True)
    raw = generate_dataset() if df is None else df.copy()
    raw["event_date"] = pd.to_datetime(raw["event_date"])
    audit = structural_audit(raw)
    profile_columns(raw).to_csv(OUTPUT / "column_profile.csv", index=False)
    numeric_profile(raw).to_csv(OUTPUT / "numeric_profile.csv")
    raw.isna().mean().sort_values(ascending=False).rename("missing_pct").to_csv(OUTPUT / "missingness.csv")

    clean = raw.drop_duplicates().copy()
    clean["device"] = clean["device"].fillna("Missing")
    clean["income_missing"] = clean["household_income"].isna().astype(int)
    clean["household_income"] = clean.groupby(["region", "channel"])["household_income"].transform(lambda s: s.fillna(s.median()))
    clean["engagement_score"] = clean["engagement_score"].fillna(clean["engagement_score"].median())
    clean["month"] = clean["event_date"].dt.to_period("M").astype(str)
    clean["revenue_per_session"] = clean["revenue"] / clean["sessions"].clip(lower=1)
    clean["log_income"] = np.log1p(clean["household_income"])

    channel = clean.groupby("channel", as_index=False).agg(customers=("customer_id","count"), spend=("media_spend","sum"), leads=("became_lead","sum"), qualified=("qualified_lead","sum"), conversions=("converted","sum"), revenue=("revenue","sum"))
    channel["lead_rate"] = channel.leads / channel.customers
    channel["qualification_rate"] = channel.qualified / channel.leads.clip(lower=1)
    channel["conversion_rate"] = channel.conversions / channel.customers
    channel["roas"] = channel.revenue / channel.spend.replace(0, np.nan)
    channel.to_csv(OUTPUT / "channel_performance.csv", index=False)

    monthly = clean.groupby("month", as_index=False).agg(customers=("customer_id","count"), conversions=("converted","sum"), revenue=("revenue","sum"))
    monthly["conversion_rate"] = monthly.conversions / monthly.customers
    monthly["rolling_conversion_rate"] = monthly.conversion_rate.rolling(3, min_periods=1).mean()
    monthly["anomaly_z"] = (monthly.conversion_rate - monthly.conversion_rate.rolling(6, min_periods=3).mean()) / monthly.conversion_rate.rolling(6, min_periods=3).std()
    monthly.to_csv(OUTPUT / "monthly_trends.csv", index=False)

    pearson = clean[NUMERIC + ["converted"]].corr(method="pearson")
    spearman = clean[NUMERIC + ["converted"]].corr(method="spearman")
    pearson.to_csv(OUTPUT / "pearson_correlations.csv")
    spearman.to_csv(OUTPUT / "spearman_correlations.csv")
    associations = pd.DataFrame([{"variable_1":a, "variable_2":b, "cramers_v":cramers_v(clean[a], clean[b])} for i,a in enumerate(CATEGORICAL) for b in CATEGORICAL[i+1:]])
    associations.to_csv(OUTPUT / "categorical_associations.csv", index=False)
    tests = statistical_tests(clean)
    tests.to_csv(OUTPUT / "statistical_tests.csv", index=False)
    scored, clusters = multivariate_analysis(clean)
    clusters.to_csv(OUTPUT / "cluster_summary.csv", index=False)

    px.histogram(clean, x="engagement_score", color="converted", marginal="box", barmode="overlay", title="Engagement distribution by conversion").write_html(OUTPUT / "univariate_bivariate.html")
    px.imshow(spearman, text_auto=".2f", color_continuous_scale="RdBu", zmin=-1, zmax=1, title="Spearman correlation matrix").write_html(OUTPUT / "correlation_heatmap.html")
    px.scatter(scored.sample(min(2500, len(scored)), random_state=42), x="pc1", y="pc2", color="cluster", symbol="converted", title="PCA view of exploratory customer segments").write_html(OUTPUT / "pca_clusters.html")
    funnel = go.Figure(go.Funnel(y=["Visitors", "Leads", "Qualified", "Converted"], x=[len(clean), clean.became_lead.sum(), clean.qualified_lead.sum(), clean.converted.sum()]))
    funnel.update_layout(title="Marketing funnel")
    funnel.write_html(OUTPUT / "funnel.html")

    summary = {"audit": audit, "clean_rows": len(clean), "pearson_engagement_conversion": round(float(pearson.loc["engagement_score", "converted"]), 4), "strongest_channel": channel.sort_values("conversion_rate", ascending=False).iloc[0]["channel"], "significant_tests": tests.loc[tests.p_value < .05, "test"].tolist(), "artifacts": sorted(path.name for path in OUTPUT.iterdir())}
    (OUTPUT / "executive_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(run_eda(), indent=2))


