import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold
import joblib
from tqdm import tqdm
import os

# Ensure the 'results' folder exists
os.makedirs("results", exist_ok=True)

# Load the labeled training data from CSV
data = pd.read_csv("./classifications/xgb_test_predictions_with_embeddings.csv", header=None)
data = data.values  # Convert DataFrame to NumPy array

# Assuming the last column is the label and the rest are features
train_embeddings = data[:, :-1]
train_labels = data[:, -1].astype(int) - 1  # 0-based labels

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    train_embeddings, train_labels, test_size=0.2, random_state=42
)

# Define the Random Forest model with the best parameters
rf_model = RandomForestClassifier(
    n_estimators=200, 
    max_depth=20, 
    min_samples_split=5, 
    random_state=42, 
    n_jobs=-1
)

# Cross-validation with tqdm progress bar
cv = StratifiedKFold(n_splits=5)
cross_val_scores = []

for train_index, val_index in tqdm(cv.split(X_train, y_train), total=cv.get_n_splits(), desc="Cross-validation"):
    X_cv_train, X_cv_val = X_train[train_index], X_train[val_index]
    y_cv_train, y_cv_val = y_train[train_index], y_train[val_index]
    rf_model.fit(X_cv_train, y_cv_train)
    val_predictions = rf_model.predict(X_cv_val)
    score = accuracy_score(y_cv_val, val_predictions)
    cross_val_scores.append(score)

mean_cv_score = np.mean(cross_val_scores)
std_cv_score = np.std(cross_val_scores)

# Save cross-validation results to a text file in the 'results' folder
with open("results/RandomForestResults.txt", "w") as f:
    f.write(f"Cross-validation mean accuracy: {mean_cv_score}\n")
    f.write(f"Cross-validation accuracy std: {std_cv_score}\n")

# Train the model on the full training data
rf_model.fit(X_train, y_train)

# Save the trained model
joblib.dump(rf_model, "./models/random_forest_model.joblib")
print("Model training completed and saved.")

# Evaluate the model on the validation set
val_predictions = rf_model.predict(X_val)

# Adjust labels for reporting
y_val += 1
val_predictions += 1

# Evaluate predictions
val_accuracy = accuracy_score(y_val, val_predictions)
val_f1 = classification_report(y_val, val_predictions, output_dict=True)["weighted avg"]["f1-score"]
conf_matrix = confusion_matrix(y_val, val_predictions)

# Save evaluation results to the text file in the 'results' folder
with open("results/RandomForestResults.txt", "a") as f:
    f.write(f"\nValidation Set Accuracy: {val_accuracy}\n")
    f.write(f"Validation Set F1 Score: {val_f1}\n")
    f.write(f"Validation Set Classification Report:\n{classification_report(y_val, val_predictions)}\n")
    f.write(f"Confusion Matrix:\n{conf_matrix}\n")

# Load the test data without labels
test_data = np.load("./data/split/Pca_testSet.npz")
test_features = test_data["data"]  # Features only

# Ensure feature count matches the model's expected input
if test_features.shape[1] != X_train.shape[1]:
    raise ValueError("Feature count mismatch between training and test data.")

# Predict using the Random Forest model
test_predictions = rf_model.predict(test_features)
test_predictions += 1  # Adjust predictions to original labeling

# Combine test features with predictions
test_results = np.hstack((test_features, test_predictions.reshape(-1, 1)))

# Save predictions and embeddings to a CSV file
np.savetxt("results/rf_test_predictions_with_embeddings.csv", test_results, delimiter=",", fmt="%.6f")
print("Predictions and embeddings for the test set have been saved to 'results/rf_test_predictions_with_embeddings.csv'.")
