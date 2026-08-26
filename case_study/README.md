# Comprehensive Python EDA Case Study

## Business problem

A marketing team needs to understand customer acquisition quality, funnel leakage, channel efficiency, behavioral drivers, and segments before choosing experiments or predictive models. This case study performs a complete exploratory analysis on a deterministic synthetic customer-level marketing funnel dataset containing realistic missing values, duplicates, skew, structural zeros, and extreme observations.

## Reproduce the analysis

```bash
pip install -r requirements.txt
python generate_data.py
python eda_case_study.py
pytest -q
```

All tables, diagnostics, charts, and the machine-readable executive summary are written to `outputs/`.

## EDA workflow and techniques

### 1. Frame the analytical question

- Define the business decision, observation grain, population, time window, and metric denominator.
- Separate descriptive patterns from causal conclusions.
- Identify target outcomes without using future information or leakage variables.

### 2. Structural data audit

- Dataset dimensions, memory use, date coverage, data types, and sample values
- Row and entity-key uniqueness
- Duplicate detection
- Cardinality and constant/high-cardinality columns
- Funnel invariants: `converted ≤ qualified ≤ lead`

### 3. Missing-data analysis

- Missing counts and percentages by column
- Structural versus unexpected missingness
- Missingness indicators as potential behavioral signals
- Group-median imputation for income and robust median imputation for engagement
- Explicit `Missing` category for unknown devices

Imputation is performed only after the missingness mechanism is considered. The raw data remains unchanged.

### 4. Univariate analysis

- Count, mean, standard deviation, min/max, and percentiles
- Median, IQR, skewness, and kurtosis
- Histograms, marginal boxplots, and categorical frequency tables
- Log transformation for right-skewed income
- Structural zeros in spend and revenue

### 5. Outlier investigation

- Tukey IQR fences
- Median absolute deviation and robust z-scores
- Domain validation before exclusion
- Comparison of extreme values across channel and conversion outcomes

Outliers are flagged rather than automatically removed because exceptional spend or revenue may represent valid high-value cases.

### 6. Bivariate relationships

- Pearson correlation for linear relationships
- Spearman correlation for monotonic and rank relationships
- Cross-tabs and normalized conversion rates
- Cramér’s V for categorical associations
- Distribution comparisons by outcome
- Segment-level funnel and performance ratios

### 7. Statistical hypothesis checks

- Welch’s t-test for unequal-variance mean comparisons
- Mann–Whitney U for nonparametric rank comparisons
- Chi-square independence test for categorical association
- Kruskal–Wallis test across multiple channel distributions

P-values are paired with practical effect sizes and business context; statistical significance alone is not treated as importance.

### 8. Multivariate exploration

- Standardization and one-hot encoding
- Principal component analysis for low-dimensional structure
- K-means as an exploratory segmentation aid
- Cluster profiles by conversion and revenue
- Correlation review for redundancy and multicollinearity risk

Clusters are descriptive hypotheses, not automatically valid customer personas.

### 9. Time-series and anomaly exploration

- Monthly aggregation at a declared grain
- Rolling conversion-rate baseline
- Rolling z-score anomaly flag
- Trend, seasonality, mix-shift, and tracking-change considerations

### 10. Funnel, cohort, and segment analysis

- Visitor → lead → qualified lead → conversion funnel
- Channel conversion, qualification, ROAS, and revenue comparisons
- Region and device slices
- Denominator-safe rate calculations
- Checks for Simpson’s paradox and survivorship bias

### 11. Feature engineering for exploration

- Calendar periods
- Revenue per session
- Log income
- Missingness indicator
- Funnel-stage rates and efficiency metrics

Engineered features are documented and kept separate from raw fields.

### 12. Communication and reproducibility

- CSV profiles for auditability
- Interactive HTML charts for exploration
- JSON executive summary for downstream automation
- Deterministic random seed
- Automated tests for quality issues and artifact generation
- Clear boundary between observation, inference, and causal claim

## Deliverables

| Artifact | Purpose |
|---|---|
| `column_profile.csv` | Schema, cardinality, and missingness audit |
| `numeric_profile.csv` | Distribution, percentile, skew, kurtosis, and outlier profile |
| `missingness.csv` | Missing-data ranking |
| `pearson_correlations.csv` | Linear relationships |
| `spearman_correlations.csv` | Monotonic relationships |
| `categorical_associations.csv` | Cramér’s V relationships |
| `statistical_tests.csv` | Hypothesis-check results |
| `channel_performance.csv` | Funnel and channel efficiency |
| `monthly_trends.csv` | Trend and anomaly diagnostics |
| `cluster_summary.csv` | Multivariate exploratory segments |
| HTML charts | Distribution, funnel, correlation, and PCA visuals |
| `executive_summary.json` | Decision-ready machine-readable summary |

## Decision framework

Material findings should end in one of five outcomes: a tracking fix, business decision, controlled experiment, model feature, or explicit no-action conclusion. EDA discovers patterns and generates hypotheses; it does not establish causality by itself.

