"""
SHAP analysis for the 60-month beta prediction models.

The script reproduces the model specifications used for the SHAP analysis and
writes figures and summary tables to outputs/shap_60m/. Input data are not
included in this repository.
"""

from pathlib import Path
import json
import warnings

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET = "future_beta_60m_w"
TARGET_WEEKS = "n_weeks_target_60m"
MIN_WEEKS = 260

DEV_FROM, DEV_TO = 2015, 2017
TEST_FROM, TEST_TO = 2018, 2019
SEED = 42

ID_COL = "orbis_id"
YEAR_COL = "fiscal_year"
COUNTRY_COL = "country_iso"
CUTOFF_COL = "cutoff_date"
RAW_BETA_COL = "hist_beta_60m_w"
SHRINK_COL = "hist_beta_60m_w_shrink1"

CORE = [
    "log_assets", "roa", "roe", "ebit_margin", "ebitda_margin",
    "asset_turnover", "current_ratio", "working_capital_to_assets",
    "lt_debt_to_assets", "lt_debt_to_equity", "tangible_assets_share",
]
BETAS = [
    "hist_beta_12m_w", "hist_beta_36m_w", "hist_beta_60m_w",
    "hist_beta_12m_w_shrink1", "hist_beta_36m_w_shrink1", "hist_beta_60m_w_shrink1",
]
CATS = ["country_iso", "nace4"]

VARIANTS = {
    "core": CORE + CATS,
    "core_plus_betas": CORE + BETAS + CATS,
}

MLP_HP = {
    "hidden_layer_sizes": (64, 32),
    "alpha": 0.001,
    "learning_rate_init": 0.0005,
}

FIRMS = [
    {"match": "ORANGE", "country": "FR", "year": 2018},
    {"match": "RENAULT", "country": "FR", "year": 2018},
    {"match": "MERCEDES", "country": "DE", "year": 2019},
]

BASE_DIR = Path(__file__).resolve().parent
DATA = BASE_DIR / "data"
OUT = BASE_DIR / "outputs" / "shap_60m"
OUT.mkdir(parents=True, exist_ok=True)

PANEL_PATH = DATA / "main_data_branch_60m.csv"
MACRO_PATH = DATA / "macro_country_fiscal_year_final.csv"
REQUIRED_INPUTS = [PANEL_PATH, MACRO_PATH]

missing_inputs = [str(path) for path in REQUIRED_INPUTS if not path.exists()]
if missing_inputs:
    raise FileNotFoundError(
        "Missing input files. Place the required CSV files in the data/ directory:\n"
        + "\n".join(missing_inputs)
    )

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
})

FEATURE_LABELS = {
    "log_assets": "Velikost firmy (log aktiv)",
    "roa": "ROA",
    "roe": "ROE",
    "ebit_margin": "EBIT marže",
    "ebitda_margin": "EBITDA marže",
    "asset_turnover": "Obrat aktiv",
    "current_ratio": "Běžná likvidita",
    "working_capital_to_assets": "Pracovní kapitál / aktiva",
    "lt_debt_to_assets": "Dlouhodobý dluh / aktiva",
    "lt_debt_to_equity": "Dlouhodobý dluh / vlastní kapitál",
    "tangible_assets_share": "Podíl stálých aktiv",
    "hist_beta_12m_w": "Historická beta 12M",
    "hist_beta_36m_w": "Historická beta 36M",
    "hist_beta_60m_w": "Historická beta 60M",
    "hist_beta_12m_w_shrink1": "Historická beta 12M, shrinkage",
    "hist_beta_36m_w_shrink1": "Historická beta 36M, shrinkage",
    "hist_beta_60m_w_shrink1": "Historická beta 60M, shrinkage",
    "inflation_yoy_cutoff": "Inflace",
    "policy_rate_cutoff": "Měnověpolitická sazba",
    "gdp_growth_lag1": "Růst HDP",
}

COUNTRY_LABELS = {
    "AT": "Rakousko", "BE": "Belgie", "CH": "Švýcarsko", "DE": "Německo",
    "DK": "Dánsko", "ES": "Španělsko", "FI": "Finsko", "FR": "Francie",
    "GB": "Spojené království", "IE": "Irsko", "IT": "Itálie", "NL": "Nizozemsko",
    "NO": "Norsko", "PT": "Portugalsko", "SE": "Švédsko",
}

NACE_LABELS = {
    "2120": "Farmaceutický průmysl",
    "2611": "Elektronické součástky",
    "2651": "Měřicí a navigační přístroje",
    "2910": "Výroba motorových vozidel",
    "3250": "Zdravotnické přístroje",
    "3511": "Výroba elektřiny",
    "4771": "Maloobchod s oděvy",
    "6110": "Telekomunikační služby",
    "6190": "Telekomunikační služby",
    "6201": "Programování",
    "6420": "Holdingové společnosti",
    "7010": "Vedení podniků",
    "7022": "Manažerské poradenství",
}


def feature_label(name):
    if name in FEATURE_LABELS:
        return FEATURE_LABELS[name]
    if name.startswith("country_iso_"):
        code = name.replace("country_iso_", "")
        return "Země: " + COUNTRY_LABELS.get(code, code)
    if name.startswith("nace4_"):
        code = name.replace("nace4_", "").replace(".0", "")
        return "Odvětví: " + NACE_LABELS.get(code, code)
    return name


def feature_group(name):
    if name.startswith("country_iso_"):
        return "Země"
    if name.startswith("nace4_"):
        return "Odvětví"
    if name in BETAS:
        return "Historické bety"
    return "Firemní ukazatele"


def group_color(name):
    colors = {
        "Firemní ukazatele": "#2A4F7C",
        "Historické bety": "#7A4060",
        "Země": "#2A6B5A",
        "Odvětví": "#C8762E",
    }
    return colors.get(feature_group(name), "#777777")


def plot_importance(features, values, out_path, max_display=20):
    imp = sorted(zip(features, values), key=lambda item: item[1], reverse=True)[:max_display]
    labels = [feature_label(name) for name, _ in imp][::-1]
    vals = [value for _, value in imp][::-1]
    cols = [group_color(name) for name, _ in imp][::-1]

    plt.figure(figsize=(10, 7))
    plt.barh(labels, vals, color=cols)
    plt.xlabel("Průměrná absolutní SHAP hodnota")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def make_preprocessor(features):
    num = [col for col in features if col in set(CORE + BETAS)]
    cat = [col for col in features if col in set(CATS)]
    num_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([("num", num_pipe, num), ("cat", cat_pipe, cat)])


def transformed_feature_names(preprocessor, features):
    num = [col for col in features if col in set(CORE + BETAS)]
    cat = [col for col in features if col in set(CATS)]
    cat_names = preprocessor.named_transformers_["cat"].named_steps["oh"].get_feature_names_out(cat)
    return num + list(cat_names)


def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"mae": float(mae), "rmse": float(rmse), "r2": float(r2)}


def load_panel(path, label):
    df = pd.read_csv(path)
    df.columns = [str(col).strip() for col in df.columns]

    string_columns = [ID_COL, COUNTRY_COL, "company_name", "isin", "nace4"]
    for col in string_columns:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip()
            df[col] = df[col].replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "<NA>": pd.NA})

    df[ID_COL] = pd.to_numeric(df[ID_COL], errors="coerce").astype("Int64")
    df[YEAR_COL] = pd.to_numeric(df[YEAR_COL], errors="coerce").astype("Int64")
    df[COUNTRY_COL] = df[COUNTRY_COL].astype("string").str.strip().str.upper()
    df[CUTOFF_COL] = pd.to_datetime(df[CUTOFF_COL], errors="coerce")
    df[TARGET_WEEKS] = pd.to_numeric(df[TARGET_WEEKS], errors="coerce")

    exempt = set(string_columns) | {CUTOFF_COL}
    for col in df.columns:
        if col not in exempt:
            df[col] = pd.to_numeric(df[col], errors="ignore")

    df = df.dropna(
        subset=[ID_COL, YEAR_COL, COUNTRY_COL, CUTOFF_COL, TARGET, RAW_BETA_COL, SHRINK_COL, TARGET_WEEKS]
    ).copy()

    if df.duplicated([ID_COL, YEAR_COL]).sum() > 0:
        raise RuntimeError(f"{label}: duplicate firm-year rows")

    df["dataset"] = label
    return df


def load_macro(path):
    macro_cols = [COUNTRY_COL, YEAR_COL, CUTOFF_COL, "inflation_yoy_cutoff", "policy_rate_cutoff", "gdp_growth_lag1"]
    df = pd.read_csv(path)
    df.columns = [str(col).strip() for col in df.columns]
    df[COUNTRY_COL] = df[COUNTRY_COL].astype("string").str.strip().str.upper()
    df[YEAR_COL] = pd.to_numeric(df[YEAR_COL], errors="coerce").astype("Int64")
    df[CUTOFF_COL] = pd.to_datetime(df[CUTOFF_COL], errors="coerce")

    for col in ["inflation_yoy_cutoff", "policy_rate_cutoff", "gdp_growth_lag1"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=macro_cols).copy()
    if df.duplicated([COUNTRY_COL, YEAR_COL]).sum() > 0:
        raise RuntimeError("macro: duplicate country-year rows")
    return df


def merge_macro(panel_df, macro_df, label):
    merged = panel_df.merge(
        macro_df,
        on=[COUNTRY_COL, YEAR_COL],
        how="left",
        validate="many_to_one",
        suffixes=("", "_macro"),
    )

    if merged["inflation_yoy_cutoff"].isna().any():
        raise RuntimeError(f"{label}: missing macro rows after merge")

    macro_cutoff_col = f"{CUTOFF_COL}_macro"
    if macro_cutoff_col in merged.columns:
        mismatch = merged[merged[CUTOFF_COL] != merged[macro_cutoff_col]]
        if len(mismatch) > 0:
            raise RuntimeError(f"{label}: panel and macro cutoff_date mismatch")

    drop_cols = [col for col in [macro_cutoff_col, "macro_reference_month_end"] if col in merged.columns]
    if drop_cols:
        merged = merged.drop(columns=drop_cols)

    return merged


def load_data():
    panel = load_panel(PANEL_PATH, "main_data_branch_60m")
    macro = load_macro(MACRO_PATH)
    df = merge_macro(panel, macro, "main_data_branch_60m")
    df = df[df[TARGET_WEEKS] >= MIN_WEEKS].copy()
    train = df[df[YEAR_COL].between(DEV_FROM, DEV_TO)].copy()
    test = df[df[YEAR_COL].between(TEST_FROM, TEST_TO)].copy()
    return train, test


def find_firm(test_df, match, country, year):
    mask = (
        test_df["company_name"].astype("string").str.upper().str.contains(match, na=False)
        & (test_df[COUNTRY_COL] == country)
        & (test_df[YEAR_COL] == year)
    )
    return int(mask.idxmax()) if mask.any() else None


def run_variant(name, features, train_df, test_df):
    print(f"\n{name} ({len(features)} features)")

    preprocessor = make_preprocessor(features)
    x_train = preprocessor.fit_transform(train_df[features])
    x_test = preprocessor.transform(test_df[features])
    y_train = pd.to_numeric(train_df[TARGET], errors="coerce").to_numpy()
    y_test = pd.to_numeric(test_df[TARGET], errors="coerce").to_numpy()
    feature_names = transformed_feature_names(preprocessor, features)

    model = MLPRegressor(
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=12,
        random_state=SEED,
        **MLP_HP,
    )
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    model_metrics = metrics(y_test, y_pred)
    print(f"MAE={model_metrics['mae']:.4f}  RMSE={model_metrics['rmse']:.4f}  R2={model_metrics['r2']:.4f}")

    rng = np.random.RandomState(SEED)
    background_idx = rng.choice(x_train.shape[0], size=min(50, x_train.shape[0]), replace=False)
    background = x_train[background_idx]

    test_reset = test_df.reset_index(drop=True)
    firm_idx = []
    for firm in FIRMS:
        idx = find_firm(test_reset, firm["match"], firm["country"], firm["year"])
        if idx is not None:
            firm_idx.append(idx)

    sample_n = min(400, x_test.shape[0])
    rng_sample = np.random.RandomState(SEED + 1)
    random_idx = rng_sample.choice(x_test.shape[0], size=sample_n, replace=False)
    sample_idx = np.unique(np.concatenate([random_idx, np.array(firm_idx, dtype=int)]))
    x_sample = x_test[sample_idx]
    print(f"SHAP sample: {len(sample_idx)} of {x_test.shape[0]} test rows")

    explainer = shap.PermutationExplainer(model.predict, background, max_evals=2 * x_test.shape[1] + 1)
    shap_values = explainer(x_sample).values

    human_feature_names = [feature_label(name) for name in feature_names]
    sample_pos = {test_index: pos for pos, test_index in enumerate(sample_idx)}
    baseline = float(model.predict(background).mean())

    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        x_sample,
        feature_names=human_feature_names,
        max_display=20,
        show=False,
        plot_size=(10, 8),
    )
    plt.tight_layout()
    plt.savefig(OUT / f"shap_summary_{name}.png", dpi=150, bbox_inches="tight")
    plt.close()

    importance_values = np.abs(shap_values).mean(axis=0)
    plot_importance(feature_names, importance_values, OUT / f"shap_importance_{name}.png", max_display=20)

    for firm in FIRMS:
        idx = find_firm(test_reset, firm["match"], firm["country"], firm["year"])
        if idx is None or idx not in sample_pos:
            continue
        pos = sample_pos[idx]
        explanation = shap.Explanation(
            values=shap_values[pos],
            base_values=baseline,
            data=x_sample[pos],
            feature_names=human_feature_names,
        )
        plt.figure(figsize=(10, 7))
        shap.waterfall_plot(explanation, max_display=12, show=False)
        plt.tight_layout()
        plt.savefig(OUT / f"shap_waterfall_{firm['match'].lower()}_{name}.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"{firm['match']} {firm['year']}: actual={y_test[idx]:.3f} pred={y_pred[idx]:.3f}")

    meta = test_reset.iloc[sample_idx].reset_index(drop=True)
    shap_df = pd.DataFrame(shap_values, columns=feature_names)
    shap_df.insert(0, "fiscal_year", meta[YEAR_COL].values)
    shap_df.insert(0, "country_iso", meta[COUNTRY_COL].values)
    shap_df.insert(0, "company_name", meta["company_name"].values)
    shap_df.insert(0, "y_true", y_test[sample_idx])
    shap_df.insert(0, "y_pred", y_pred[sample_idx])
    shap_df.to_csv(OUT / f"shap_values_{name}.csv", index=False)

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": importance_values,
        "category": [feature_group(name) for name in feature_names],
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    importance_df.to_csv(OUT / f"shap_importance_{name}.csv", index=False)

    return model_metrics


def main():
    train_df, test_df = load_data()
    print(f"train: {len(train_df)} rows ({DEV_FROM}-{DEV_TO})")
    print(f"test:  {len(test_df)} rows ({TEST_FROM}-{TEST_TO})")

    results = {}
    for name, features in VARIANTS.items():
        results[name] = run_variant(name, features, train_df, test_df)

    with open(OUT / "shap_metrics.json", "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)

    print(f"\nOutput directory: {OUT}")


if __name__ == "__main__":
    main()
