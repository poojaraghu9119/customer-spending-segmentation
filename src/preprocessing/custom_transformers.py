# Importing the required libraries

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Defining the customer transformer class to cap the outliers of the required columns.
class PercentileCapper(BaseEstimator, TransformerMixin):
    """This custom transformer caps the outliers of the recency, frequency, avg_items_per_order, 
       and customer_lifespan_days with their 99th percentile values."""
    def __init__(self, columns, percentile=0.99):
        self.columns = columns
        self.percentile = percentile
        self.cap_values_ = {}

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        for col in self.columns:
            self.cap_values_[col] = X[col].quantile(self.percentile)
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for col in self.columns:
            X[col] = np.where(
                X[col] > self.cap_values_[col],
                self.cap_values_[col],
                X[col]
            )
        return X







