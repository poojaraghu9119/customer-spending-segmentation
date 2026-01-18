# Importing the required libraries
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PowerTransformer, StandardScaler

# Importing the custom transformer PercentileCapper
from src.preprocessing.custom_transformers import PercentileCapper

# Importing the raw dataset
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "raw"
df = pd.read_csv(DATA_DIR / "raw.csv")

# Importing the feature_eng function and loading the dataset with the new features
from src.features.feature_engineering import feature_eng
rfm_df = feature_eng(df)

# Deriving the final dataset from this:
final_df = rfm_df.drop(["first_purchase_date", "last_purchase_date", "last_purchase"], axis = 1)

# Saving rfm_df and final_df to the processed folder of the data folder.
BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

rfm_path = PROCESSED_DIR / "rfm_features.csv"
final_path = PROCESSED_DIR / "final_features.csv"

rfm_df.to_csv(rfm_path, index=False)
final_df.to_csv(final_path, index=False)

print("Saved rfm_features.csv and final_features.csv to data/processed/")

# Further exploration like checking the correlation between the features and checking outliers using boxplots were done in the 
# EDA notebook. Based on the insights from those plots, it is found out that the features monetary, frequency, average_order_value,
# and customer_lifespan_days were highly correlated with the other features. So, they have to be removed. And the feature
# recency can also be dropped, because we have more strong features like inter_purchase_gap_mean, inter_purchase_gap_std.

final_df = final_df.drop(["monetary", "recency", "frequency", "customer_lifespan_days", "average_order_value"], axis = 1)

# Saving the file again in the same path
final_df.to_csv(final_path, index=False)

# As discussed in the notebook, all features except user_id have outliers. But the outlier seems to be valid.
# So, we can just transform the features using yeo-johnson method, since there are zeroes in inter_purchase_gap_std column, and 
# then scale all the features.

# Separating the user_id and the remaining features of the final_df:
id_col = ["user_id"]

feature_cols = ["avg_items_per_order", "active_months", "monetary_per_active_month", "frequency_per_active_month",
                "inter_purchase_gap_mean", "inter_purchase_gap_std"]

# Defining the preprocessing pipeline
feature_pipeline = Pipeline(steps=[("yeo_johnson", PowerTransformer(method="yeo-johnson", standardize=False)),
                                   ("scaler", StandardScaler())])

# Applying the pipeline to the features:
preprocessor = ColumnTransformer(transformers=[("features", feature_pipeline, feature_cols)],
                                 remainder="passthrough"  # keeps user_id unchanged
                                 )
X_processed = preprocessor.fit_transform(final_df)

# Converting it back to a dataframe
processed_feature_names = feature_cols+ id_col 

processed_df = pd.DataFrame(X_processed, columns=processed_feature_names)

# Save the processed dataset
scaled_path = PROCESSED_DIR / "processed_features.csv"
processed_df.to_csv(scaled_path, index=False)

print("Scaled features saved to data/processed/processed_features.csv")