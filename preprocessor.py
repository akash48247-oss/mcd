"""
Data preprocessing utilities for MCD model
Save preprocessor.pkl using this module if needed for consistent feature scaling
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

class MCDPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = [
            'Energy (kCal)', 'Protein (g)', 'Total fat (g)', 'Sat Fat (g)',
            'Trans fat (g)', 'Cholesterols (mg)', 'Total carbohydrate (g)',
            'Total Sugars (g)', 'Added Sugars (g)', 'Sodium (mg)'
        ]
    
    def fit(self, X):
        """Fit the preprocessor on training data"""
        self.scaler.fit(X)
        return self
    
    def transform(self, X):
        """Transform data using fitted preprocessor"""
        return self.scaler.transform(X)
    
    def fit_transform(self, X):
        """Fit and transform data"""
        return self.scaler.fit_transform(X)
    
    def inverse_transform(self, X):
        """Inverse transformation"""
        return self.scaler.inverse_transform(X)
    
    def save(self, filepath):
        """Save preprocessor to file"""
        joblib.dump(self, filepath)
    
    @staticmethod
    def load(filepath):
        """Load preprocessor from file"""
        return joblib.load(filepath)


# Example usage:
if __name__ == "__main__":
    # Create and fit preprocessor
    preprocessor = MCDPreprocessor()
    
    # Example data
    sample_data = np.array([
        [402.05, 10.24, 13.83, 5.34, 0.16, 2.49, 56.54, 7.90, 4.49, 706.13],
        [339.52, 8.50, 11.31, 4.27, 0.20, 1.47, 50.27, 7.05, 4.07, 545.34]
    ])
    
    # Fit on sample data
    preprocessor.fit(sample_data)
    
    # Save to file
    preprocessor.save("preprocessor.pkl")
    print("✅ Preprocessor saved to preprocessor.pkl")
