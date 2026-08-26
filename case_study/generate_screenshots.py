"""Generate publication-ready PNG screenshots for the EDA walkthrough."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from generate_data import generate_dataset

ROOT = Path(__file__).parent
IMAGES = ROOT / "images"
COLORS = ["#22c55e", "#60a5fa", "#f59e0b", "#f43f5e", "#a78bfa"]


def style() -> None:
    plt.rcParams.update({"figure.facecolor":"#080b12", "axes.facecolor":"#111827", "axes.edgecolor":"#334155", "axes.labelcolor":"#e5e7eb", "text.color":"#e5e7eb", "xtick.color":"#cbd5e1", "ytick.color":"#cbd5e1", "grid.color":"#334155", "font.size":10})


def save(name: str) -> None:
    plt.tight_layout()
    plt.savefig(IMAGES / name, dpi=170, bbox_inches="tight", facecolor="#080b12")
    plt.close()


def generate() -> list[str]:
    IMAGES.mkdir(exist_ok=True)
    style()
    df = generate_dataset()
    df["event_date"] = pd.to_datetime(df["event_date"])
    clean = df.drop_duplicates().copy()
    clean["device"] = clean.device.fillna("Missing")
    clean["engagement_score"] = clean.engagement_score.fillna(clean.engagement_score.median())
    clean["household_income"] = clean.groupby(["region", "channel"])["household_income"].transform(lambda s: s.fillna(s.median()))

    # 1. Workflow
    fig, ax = plt.subplots(figsize=(13, 3.2)); ax.axis("off")
    labels = ["1. Frame", "2. Audit", "3. Clean", "4. Explore", "5. Test", "6. Segment", "7. Communicate"]
    for i, label in enumerate(labels):
        x = .03 + i * .14
        ax.text(x, .5, label, transform=ax.transAxes, ha="center", va="center", fontsize=11, weight="bold", bbox=dict(boxstyle="round,pad=.65", fc="#17243a", ec=COLORS[i % len(COLORS)], lw=2))
        if i < len(labels)-1: ax.annotate("", xy=(x+.105,.5), xytext=(x+.07,.5), xycoords=ax.transAxes, arrowprops=dict(arrowstyle="->", color="#94a3b8", lw=2))
    ax.set_title("End-to-end exploratory data analysis workflow", fontsize=17, weight="bold", pad=20)
    save("01_eda_workflow.png")

    # 2. Missingness
    missing = df.isna().mean().sort_values(ascending=True)
    plt.figure(figsize=(11, 6)); plt.barh(missing.index, missing.values * 100, color="#60a5fa"); plt.xlabel("Missing values (%)"); plt.title("Missingness audit by field", fontsize=16, weight="bold"); plt.grid(axis="x", alpha=.35)
    save("02_missingness_audit.png")

    # 3. Distributions and outliers
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    axes[0].hist(clean.engagement_score, bins=30, color="#22c55e", alpha=.85); axes[0].set_title("Engagement distribution"); axes[0].set_xlabel("Engagement score"); axes[0].set_ylabel("Customers")
    axes[1].boxplot([clean.loc[clean.channel==c,"media_spend"] for c in clean.channel.unique()], tick_labels=clean.channel.unique(), patch_artist=True, boxprops=dict(facecolor="#60a5fa"), medianprops=dict(color="#f59e0b", linewidth=2), showfliers=True); axes[1].tick_params(axis="x", rotation=25); axes[1].set_title("Spend outliers by channel"); axes[1].set_ylabel("Media spend")
    fig.suptitle("Univariate distributions and outlier review", fontsize=16, weight="bold")
    save("03_distributions_outliers.png")

    # 4. Correlations
    fields = ["sessions","engagement_score","pages_viewed","media_spend","age","household_income","converted","revenue"]
    corr = clean[fields].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(9, 7)); image=ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(fields)), fields, rotation=45, ha="right"); ax.set_yticks(range(len(fields)), fields)
    for i in range(len(fields)):
        for j in range(len(fields)): ax.text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center", fontsize=8, color="white")
    fig.colorbar(image, ax=ax, shrink=.8); ax.set_title("Spearman correlation matrix", fontsize=16, weight="bold")
    save("04_correlation_matrix.png")

    # 5. Funnel
    funnel = [len(clean), clean.became_lead.sum(), clean.qualified_lead.sum(), clean.converted.sum()]
    labels = ["Visitors", "Leads", "Qualified", "Converted"]
    fig, ax = plt.subplots(figsize=(11, 6)); bars=ax.barh(labels[::-1], funnel[::-1], color=COLORS[:4][::-1]); ax.set_title("Marketing funnel and stage loss", fontsize=16, weight="bold"); ax.set_xlabel("Customers")
    for bar, value in zip(bars, funnel[::-1]): ax.text(value + max(funnel)*.012, bar.get_y()+bar.get_height()/2, f"{value:,}", va="center", weight="bold")
    save("05_funnel_analysis.png")

    # 6. Channel performance
    channel = clean.groupby("channel").agg(customers=("customer_id","count"), conversions=("converted","sum"), spend=("media_spend","sum"), revenue=("revenue","sum"))
    channel["conversion_rate"] = channel.conversions/channel.customers; channel["roas"] = channel.revenue/channel.spend.replace(0,np.nan)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5)); channel.conversion_rate.sort_values().plot.barh(ax=axes[0], color="#22c55e"); axes[0].set_title("Conversion rate by channel"); axes[0].set_xlabel("Conversion rate")
    channel.roas.dropna().sort_values().plot.barh(ax=axes[1], color="#a78bfa"); axes[1].set_title("ROAS by paid channel"); axes[1].set_xlabel("Return on ad spend")
    fig.suptitle("Segment and campaign performance", fontsize=16, weight="bold")
    save("06_channel_performance.png")

    # 7. Time and anomalies
    clean["month"] = clean.event_date.dt.to_period("M").dt.to_timestamp()
    monthly = clean.groupby("month").converted.agg(["sum","count"]); monthly["rate"]=monthly["sum"]/monthly["count"]; monthly["rolling"]=monthly["rate"].rolling(3,min_periods=1).mean(); monthly["std"]=monthly["rate"].rolling(6,min_periods=3).std()
    fig, ax=plt.subplots(figsize=(12,5.5)); ax.plot(monthly.index, monthly["rate"], marker="o", color="#60a5fa", label="Monthly rate"); ax.plot(monthly.index, monthly["rolling"], color="#22c55e", lw=3, label="3-month rolling baseline"); ax.fill_between(monthly.index, monthly["rolling"]-2*monthly["std"], monthly["rolling"]+2*monthly["std"], color="#f59e0b", alpha=.15, label="Approx. anomaly band"); ax.legend(); ax.set_ylabel("Conversion rate"); ax.set_title("Trend, rolling baseline, and anomaly review", fontsize=16, weight="bold")
    save("07_time_anomaly_analysis.png")

    # 8. PCA clusters
    matrix = StandardScaler().fit_transform(clean[["sessions","engagement_score","pages_viewed","media_spend","age","household_income"]])
    pca = PCA(n_components=2).fit_transform(matrix); clusters = KMeans(n_clusters=4, random_state=42, n_init=20).fit_predict(matrix)
    fig, ax=plt.subplots(figsize=(10,7))
    for cluster in range(4):
        mask=clusters==cluster; ax.scatter(pca[mask,0], pca[mask,1], s=14, alpha=.5, label=f"Cluster {cluster}", color=COLORS[cluster])
    ax.legend(); ax.set_xlabel("Principal component 1"); ax.set_ylabel("Principal component 2"); ax.set_title("PCA view of exploratory customer segments", fontsize=16, weight="bold")
    save("08_pca_clusters.png")
    return sorted(path.name for path in IMAGES.glob("*.png"))


if __name__ == "__main__":
    print("\n".join(generate()))
