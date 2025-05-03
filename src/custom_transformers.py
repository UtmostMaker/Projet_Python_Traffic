# custom_transformers.py
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """
    Encodes categorical features using their frequency (percentage)
    learned from the training data. Handles potential Categorical dtypes.
    """
    def __init__(self, cols):
        # cols: List of column names to be frequency encoded
        if not isinstance(cols, list):
            self.cols = [cols]
        else:
            self.cols = cols
        self.freq_map = {} # Dictionary to store frequency mappings

    def fit(self, X, y=None):
        """
        Learn the frequency of each category for the specified columns
        from the training data X.
        """
        # Ensure X is a DataFrame
        X_ = X.copy()
        if not isinstance(X_, pd.DataFrame):
             X_ = pd.DataFrame(X_)

        for col in self.cols:
            if col not in X_.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame")
            # Calculate normalized frequencies (percentages)
            # Convert to string first to handle mixed types or categoricals
            frequencies = X_[col].astype(str).value_counts(normalize=True)
            self.freq_map[col] = frequencies
            # Storing frequencies learned during fit.
            # Using normalize=True gives percentage, robust to dataset size.
            # Using astype(str) ensures consistent category handling.
        return self

    def transform(self, X):
        """
        Transform the categories in X to their learned frequencies.
        Handles categories not seen during fit by assigning frequency 0.
        """
        # Ensure X is a DataFrame for reliable column access
        X_copy = X.copy()
        if not isinstance(X_copy, pd.DataFrame):
             X_copy = pd.DataFrame(X_copy)


        for col in self.cols:
            if col not in X_copy.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame")

            # 1. Map categories to their learned frequencies using .astype(str)
            #    for consistent lookup, matching the fit step.
            mapped_series = (
                X_copy[col].astype(str).map(self.freq_map.get(col, {}))
            )
            # .map looks up each value in the learned freq_map.
            # .get(col, {}) prevents KeyError if col wasn't in fit (though unlikely here).
            # Using .astype(str) is crucial for consistent mapping.

            # 2. 
            # Convert the result (frequencies or NaN) explicitly to float.
            # This breaks the link to any original 'Categorical' dtype and
            # allows filling NaN with a numeric value (0).
            numeric_series = mapped_series.astype(float)
            # This astype(float) is key to prevent the
            # 'Cannot setitem on a Categorical with a new category' error,
            # by ensuring we work with a numeric series before fillna.

            # 3. Fill NaN values (for categories not seen during fit) with 0.
            # Create the new frequency column name.
            new_col_name = f'{col}_freq'
            X_copy[new_col_name] = numeric_series.fillna(0)
            # Filling NaNs with 0 assumes unseen categories
            # have zero frequency in the training context.

        # 4. Drop the original categorical columns after creating freq columns.
        # Ensure only columns present in X_copy are dropped.
        cols_to_drop = [c for c in self.cols if c in X_copy.columns]
        X_copy = X_copy.drop(columns=cols_to_drop)
        # Returning a DataFrame with original columns
        # replaced by their frequency-encoded counterparts.

        return X_copy
