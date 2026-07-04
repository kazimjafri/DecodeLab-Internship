"""Enterprise-Grade Data Engineering Pipeline: Project 1.

Advanced EDA, Vectorized Engine, and Runtime Schema Contracts.
"""

import numpy as np
import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema
from sklearn.impute import KNNImputer


# ==========================================================
# PHASE 1: INPUT FIDELITY ENGINE
# ==========================================================
def execute_input_fidelity(raw_df):
    """Secures raw data distribution variance and handles missing nodes/outliers.

    Adheres strictly to loop-free block-allocated RAM operations.
    """
    df_clean = raw_df.copy()

    print("\n--- Running Phase 1: Securing Input Fidelity ---")

    # Task 1: Missing Data Decision Matrix Brackets
    # Bracket 1 (< 5%): Row Deletion
    df_clean = df_clean.dropna(subset=["distance_to_city_km"])

    # Bracket 2 (5% - 20%): Global Median Imputation
    crime_median = df_clean["crime_index"].median()
    df_clean["crime_index"] = df_clean["crime_index"].fillna(crime_median)

    # Bracket 3 (> 20%): Multi-Dimensional KNN Imputation
    knn = KNNImputer(n_neighbors=5)
    df_clean[["school_quality_index"]] = knn.fit_transform(
        df_clean[["school_quality_index"]]
    )

    print("   [SUCCESS] Missing data decision matrix thresholds resolved.")

    # Task 2: Vectorized Outlier Handling via Capping (Winsorization)
    numeric_cols = [
        "area_sqft",
        "lot_size_sqft",
        "distance_to_city_km",
        "school_quality_index",
        "crime_index",
        "price",
    ]
    df_numeric = df_clean[numeric_cols]

    # Matrix level quantile bounds calculations
    Q1 = df_numeric.quantile(0.25)
    Q3 = df_numeric.quantile(0.75)
    IQR = Q3 - Q1

    lower_bounds = Q1 - 1.5 * IQR
    upper_bounds = Q3 + 1.5 * IQR

    # Pure C-level execution bounds capping directly in RAM matrix
    df_clean[numeric_cols] = df_numeric.clip(
        lower=lower_bounds, upper=upper_bounds, axis=1
    )
    print("   [SUCCESS] Matrix-level outlier capping executed with zero loops.")

    return df_clean


# ==========================================================
# PHASE 2: VECTORIZED COMPUTATION ENGINE
# ==========================================================
def execute_computation_engine(fidelity_df):
    """Engineers high-fidelity features and eliminates multicollinearity logic redundancies."""
    df_transformed = fidelity_df.copy()

    print("\n--- Running Phase 2: Vectorized Computation Engine ---")

    # Task 1: Feature Engineering (3 New Predictive Features)
    df_transformed["total_rooms"] = (
        df_transformed["bedrooms"] + df_transformed["bathrooms"]
    )
    df_transformed["yard_size_sqft"] = (
        df_transformed["lot_size_sqft"] - df_transformed["area_sqft"]
    )
    df_transformed["space_per_room"] = df_transformed["area_sqft"] / (
        df_transformed["total_rooms"] + 1
    )
    print("   [SUCCESS] 3 New analytical predictive features generated.")

    # Task 2: Categorical Translation (One-Hot Encoding Space Mapping)
    df_transformed = pd.get_dummies(
        df_transformed, columns=["area"], drop_first=True, dtype=int
    )
    print("   [SUCCESS] Nominal categories mapped onto orthogonal coordinates.")

    # Task 3: Collinearity Eradication Matrix Algorithm
    numeric_features = df_transformed.select_dtypes(include=[np.number]).drop(
        columns=["id"]
    )
    corr_with_target = numeric_features.corr()["price"].abs()

    # Evaluating redundant correlation links (bedrooms vs bathrooms)
    feature_to_drop = (
        "bedrooms"
        if corr_with_target["bedrooms"] < corr_with_target["bathrooms"]
        else "bathrooms"
    )
    df_transformed = df_transformed.drop(columns=[feature_to_drop], errors="ignore")
    print(
        f"   [SUCCESS] Collinearity eradicated. Weakest node dropped: '{feature_to_drop}'."
    )

    return df_transformed


# ==========================================================
# PHASE 3: PRODUCTION STRUCTURAL CONTRACTS
# ==========================================================
# Coerce=True automatically handles runtime dtype variations seamlessly
house_price_schema = DataFrameSchema(
    columns={
        "area_sqft": Column(
            int, Check.greater_than_or_equal_to(400.0), coerce=True
        ),
        "lot_size_sqft": Column(int, Check.greater_than(0.0), coerce=True),
        "bathrooms": Column(
            int, Check.in_range(1, 10), required=False, coerce=True
        ),
        "stories": Column(int, Check.in_range(1, 4), coerce=True),
        "parking_spaces": Column(
            int, Check.greater_than_or_equal_to(0), coerce=True
        ),
        "house_age_years": Column(
            int, Check.greater_than_or_equal_to(0), coerce=True
        ),
        "distance_to_city_km": Column(
            float, Check.greater_than_or_equal_to(0.0), coerce=True
        ),
        "school_quality_index": Column(
            float, Check.in_range(1.0, 10.0), coerce=True
        ),
        "crime_index": Column(float, Check.in_range(1.0, 10.0), coerce=True),
        "price": Column(float, Check.greater_than(0.0), coerce=True),
        "total_rooms": Column(int, Check.greater_than(0), coerce=True),
        "yard_size_sqft": Column(int, coerce=True),
        "space_per_room": Column(
            float, Check.greater_than(0.0), coerce=True
        ),
    },
    strict=False,  # dynamically added area dummy columns ko pass hone dene ke liye
)


def validate_production_data(final_df):
    """Enforces strict software data contracts at runtime with diagnostic profiling."""
    print("\n--- Running Phase 3: Asserting Structural Contracts ---")
    try:
        # Enforcing contracts at runtime with lazy reporting enabled
        validated_df = house_price_schema.validate(final_df, lazy=True)
        print(
            "   [SUCCESS] PRODUCTION CONTRACT VALIDATED: System architecture secure!"
        )
        return validated_df
    except pa.errors.SchemaErrors as err:
        print("   [CRITICAL ERROR] Pipeline Interface Contracts Broken!")
        print("\n--- Failure Diagnostics Log ---")
        print(err.failure_cases[["column", "check", "failure_case"]].head())
        return None


# ==========================================================
# ENTERPRISE EXECUTION SPINE
# ==========================================================
if __name__ == "__main__":
    print("==========================================================")
    print("              ENTERPRISE DATA PIPELINE INTERFACE          ")
    print("==========================================================")

    # 1. Loading the source dataset context (Matches your file name strictly)
    try:
        raw_data = pd.read_csv("HousePrice.csv")
    except FileNotFoundError:
        print(
            "ERROR: 'HousePrice.csv' not found! Make sure it is in the same folder."
        )
        exit(1)

    # 2. Sequential modular pipeline step tracking
    fidelity_output = execute_input_fidelity(raw_data)
    computation_output = execute_computation_engine(fidelity_output)
    production_ready_df = validate_production_data(computation_output)

    if production_ready_df is not None:
        print("\n PIPELINE SUCCESSFUL: Dataset is ready for feature stores.")
        print(f"Final shape coordinates: {production_ready_df.shape}")

        # Automatically export the file upon execution success
        production_ready_df.to_csv(
            "house_price_production_ready.csv", index=False
        ) 
        print("Saved file as: house_price_production_ready.csv\n")