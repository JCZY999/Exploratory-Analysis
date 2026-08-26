"""Generate a deterministic, intentionally imperfect marketing funnel dataset."""

from pathlib import Path

import numpy as np
import pandas as pd

CHANNELS = ["Paid Search", "Paid Social", "Organic", "Email", "Affiliate"]
REGIONS = ["West", "Southwest", "Midwest", "Northeast"]
DEVICES = ["Desktop", "Mobile", "Tablet"]


def generate_dataset(rows: int = 6000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.to_datetime("2024-01-01") + pd.to_timedelta(rng.integers(0, 540, rows), unit="D")
    channel = rng.choice(CHANNELS, rows, p=[.28, .24, .21, .16, .11])
    region = rng.choice(REGIONS, rows, p=[.30, .23, .27, .20])
    device = rng.choice(DEVICES, rows, p=[.42, .51, .07])
    sessions = np.maximum(1, rng.negative_binomial(3, .42, rows))
    engagement = np.clip(rng.beta(2.3, 3.2, rows), 0, 1)
    pages = np.maximum(1, rng.poisson(2.5 + 5 * engagement, rows))
    spend = np.where(np.isin(channel, ["Organic", "Email"]), 0, rng.gamma(2.2, 38, rows))
    lead_logit = -2.1 + 2.8 * engagement + .09 * pages + .35 * (channel == "Paid Search") - .25 * (device == "Mobile")
    lead_prob = 1 / (1 + np.exp(-lead_logit))
    lead = rng.random(rows) < lead_prob
    qualified_prob = np.clip(.34 + .32 * engagement + .09 * (channel == "Organic") - .07 * (region == "Southwest"), .03, .93)
    qualified = lead & (rng.random(rows) < qualified_prob)
    converted_prob = np.clip(.20 + .42 * engagement + .12 * (channel == "Email") + .07 * (device == "Desktop"), .02, .94)
    converted = qualified & (rng.random(rows) < converted_prob)
    revenue = np.where(converted, rng.lognormal(7.15 + .35 * engagement, .48, rows), 0)
    age = np.clip(rng.normal(38, 12, rows), 18, 82).round()
    income = np.clip(rng.lognormal(11.0, .52, rows), 18000, 650000)

    frame = pd.DataFrame({
        "customer_id": [f"CUST-{i:06d}" for i in range(rows)], "event_date": dates,
        "channel": channel, "region": region, "device": device, "sessions": sessions,
        "engagement_score": engagement.round(4), "pages_viewed": pages, "media_spend": spend.round(2),
        "age": age, "household_income": income.round(2), "became_lead": lead.astype(int),
        "qualified_lead": qualified.astype(int), "converted": converted.astype(int), "revenue": revenue.round(2)
    })
    # Realistic quality problems for the audit exercises.
    frame.loc[rng.choice(rows, 210, replace=False), "household_income"] = np.nan
    frame.loc[rng.choice(rows, 95, replace=False), "device"] = None
    frame.loc[rng.choice(rows, 35, replace=False), "engagement_score"] = np.nan
    frame.loc[rng.choice(rows, 12, replace=False), "media_spend"] *= 25
    duplicates = frame.sample(18, random_state=seed)
    return pd.concat([frame, duplicates], ignore_index=True)


if __name__ == "__main__":
    destination = Path(__file__).parent / "marketing_funnel.csv"
    generate_dataset().to_csv(destination, index=False)
    print(f"Wrote {destination}")


