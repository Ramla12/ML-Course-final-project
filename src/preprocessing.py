import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(file_path):
    """Load the dataset from a given file path."""
    return pd.read_csv(file_path)

def clean_data(data):
    """Clean the dataset by dropping rows with missing critical values."""
    return data.dropna(subset=['PTS', 'Age', 'MP', 'FG%', '3P%', 'FT%', 'TRB', 'AST'])

def select_features(data):
    """Select features and target variable."""
    features = ['Age', 'MP', 'FG%', '3P%', 'FT%', 'TRB', 'AST']
    target = 'PTS'
    X = data[features]
    y = (data[target] >= 10).astype(int)
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    """Split the data into training and test sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
