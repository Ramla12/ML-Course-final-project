import numpy as np
from src.preprocessing import load_data, clean_data, select_features, split_data
from src.models import train_random_forest, evaluate_model,get_feature_importance
from src.linear_regression import LinearRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, roc_auc_score, recall_score, f1_score
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Linear Regression.")
    parser.add_argument("-r", "--train", type=str, required=True, help="Path to training dataset")
    parser.add_argument("-e", "--test", type=str, required=True, help="Path to test dataset")
    parser.add_argument("-a", "--alpha", type=float, required=True, help="Learning rate")
    args = parser.parse_args()


# Load and preprocess the data
file_path = './data/nba_data_processed.csv'
data = load_data(file_path)
data_cleaned = clean_data(data)
X, y = select_features(data_cleaned)
X_train, X_test, y_train, y_test = split_data(X, y)
print("Training set shape:", X_train.shape, y_train.shape)
print("Test set shape:", X_test.shape, y_test.shape)

# Add bias term to feature set
X_train = np.c_[np.ones(X_train.shape[0]), X_train]
X_test = np.c_[np.ones(X_test.shape[0]), X_test]


# Train the Linear Regression model
lr = LinearRegression(learning_rate=0.01, max_iter=1000, tolerance=1e-4)
lr.fit(X_train, y_train)

# Evaluate the model
y_pred = lr.predict(X_test)

# Compute metrics and display results
y_test_binary = (y_test >= 0.5).astype(int)  # Threshold continuous values
y_pred_binary = (y_pred >= 0.5).astype(int)
conf_matrix = confusion_matrix(y_test_binary, y_pred_binary, labels=[0, 1])
accuracy = accuracy_score(y_test_binary, y_pred_binary)
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Training labels distribution:", np.bincount(y_train))
print("Test labels distribution:", np.bincount(y_test))

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")


# ROC-AUC
roc_auc = roc_auc_score(y_test, y_pred)
print(f"ROC-AUC: {roc_auc:.2f}")
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
feature_importance = pd.DataFrame({
    "Feature": ['Age', 'MP', 'FG%', '3P%', 'FT%', 'TRB', 'AST'],
    "Coefficient": lr.weights[1:]  # Exclude the bias term
}).sort_values(by="Coefficient", ascending=False)
print(feature_importance)


#randomforest
rf_model = train_random_forest(X_train, y_train, n_estimators=100, max_depth=None)

feature_names = ['Age', 'MP', 'FG%', '3P%', 'FT%', 'TRB', 'AST']


# Evaluate the Random Forest model
y_pred_rf, conf_matrix_rf = evaluate_model(rf_model, X_test, y_test)

'''
# Get feature importance
feature_importance_rf = get_feature_importance(rf_model, feature_names)
print("\nRandom Forest Feature Importance:")
print(feature_importance_rf)


# Plot feature importance
plt.figure(figsize=(8, 6))
plt.barh(feature_importance_rf["Feature"], feature_importance_rf["Importance"], color="skyblue")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.title("Random Forest Feature Importance")
plt.gca().invert_yaxis()
plt.show()

'''


