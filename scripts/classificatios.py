import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import joblib

# Load the labeled training data
train_data_labeled = np.load("./data/split/Pca_traningSet_labeled.npz")
train_embeddings = train_data_labeled["data"][:, :-1]  # Features
train_labels = train_data_labeled["data"][:, -1].astype(int) - 1  # 0-based labels

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    train_embeddings, train_labels, test_size=0.2, random_state=42
)

# Train the XGBoost model with regularization
xgb_model = xgb.XGBClassifier(
    objective="multi:softmax",
    use_label_encoder=False,
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    reg_alpha=0.1,
    reg_lambda=0.1,
)  # Regularization parameters
xgb_model.fit(X_train, y_train)

# Save the trained model
joblib.dump(xgb_model, "./models/xgboost_model.joblib")
print("Model training completed and saved.")

# Evaluate the model on the validation set
val_predictions = xgb_model.predict(X_val)

# Adjust labels for reporting
y_val += 1
val_predictions += 1

# Evaluate predictions
val_accuracy = accuracy_score(y_val, val_predictions)
val_f1 = classification_report(y_val, val_predictions, output_dict=True)[
    "weighted avg"
]["f1-score"]

print(f"Validation Set Accuracy: {val_accuracy}")
print(f"Validation Set F1 Score: {val_f1}")
print(
    f"Validation Set Classification Report:\n{classification_report(y_val, val_predictions)}"
)

# Load the test data without labels
test_data = np.load("./data/split/Pca_testSet.npz")
test_features = test_data["data"]  # Features only

# Ensure feature count matches the model's expected input
if test_features.shape[1] != X_train.shape[1]:
    raise ValueError("Feature count mismatch between training and test data.")

# Predict using the XGBoost model
test_predictions = xgb_model.predict(test_features)
test_predictions += 1  # Adjust predictions to original labeling

# Save predictions to a CSV file
np.savetxt("xgb_test_predictions.csv", test_predictions, delimiter=",", fmt="%d")
print("Predictions for the test set have been saved to 'xgb_test_predictions.csv'.")
