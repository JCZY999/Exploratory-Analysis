from generate_data import generate_dataset
from eda_case_study import run_eda


def test_generator_contains_intentional_quality_issues():
    data = generate_dataset(rows=800)
    assert data.duplicated().sum() > 0
    assert data.isna().sum().sum() > 0
    assert {"became_lead", "qualified_lead", "converted"}.issubset(data.columns)


def test_full_eda_produces_summary():
    summary = run_eda(generate_dataset(rows=1000))
    assert summary["audit"]["rows"] > summary["clean_rows"]
    assert summary["strongest_channel"]
    assert len(summary["artifacts"]) >= 10


