import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import joblib
from tqdm import tqdm

# Load data
data = np.load("./data/split/Pca_traningSet_labeled.npz")
combined_data = data["data"]
embeddings = combined_data[:, :-1]
labels = combined_data[:, -1]
labels = labels.astype(int) - 1

X_train, X_val, y_train, y_val = train_test_split(
    embeddings, labels, test_size=0.2, random_state=42
)

# Hyperparameter tuning for XGBoost with fewer combinations
xgb_params = {
    "n_estimators": [50, 100],
    "learning_rate": [0.01, 0.1],
    "max_depth": [3, 6],
}
model_xgb = xgb.XGBClassifier(objective="multi:softmax", use_label_encoder=False)
grid_xgb = GridSearchCV(model_xgb, xgb_params, cv=3)

# Display progress with tqdm
total_iterations = (
    len(xgb_params["n_estimators"])
    * len(xgb_params["learning_rate"])
    * len(xgb_params["max_depth"])
)
with tqdm(total=total_iterations, desc="XGBoost Hyperparameter Tuning") as pbar:
    for _ in range(total_iterations):
        grid_xgb.fit(X_train, y_train)
        pbar.update(1)

best_xgb = grid_xgb.best_estimator_
print("Best XGBoost Parameters:", grid_xgb.best_params_)
joblib.dump(best_xgb, "./models/xgboost_model.joblib")

# # Hyperparameter tuning for Random Forest with fewer combinations
# rf_params = {
#     "n_estimators": [100, 200],
#     "max_depth": [10, 20],
#     "min_samples_split": [2, 5],
# }
# model_rf = RandomForestClassifier(random_state=42)
# grid_rf = GridSearchCV(model_rf, rf_params, cv=3)

# total_iterations = (
#     len(rf_params["n_estimators"])
#     * len(rf_params["max_depth"])
#     * len(rf_params["min_samples_split"])
# )
# with tqdm(total=total_iterations, desc="Random Forest Hyperparameter Tuning") as pbar:
#     for _ in range(total_iterations):
#         grid_rf.fit(X_train, y_train)
#         pbar.update(1)

# best_rf = grid_rf.best_estimator_
# print("Best Random Forest Parameters:", grid_rf.best_params_)
# joblib.dump(best_rf, "./models/random_forest_model.joblib")

# # Naive Bayes (No hyperparameter tuning typically necessary)
# model_nb = make_pipeline(StandardScaler(), GaussianNB())
# model_nb.fit(X_train, y_train)
# joblib.dump(model_nb, "./models/naive_bayes_model.joblib")

# # Voting Classifier (using the best models found)
# voting_clf = VotingClassifier(
#     estimators=[("xgb", best_xgb), ("rf", best_rf), ("nb", model_nb)], voting="hard"
# )
# voting_clf.fit(X_train, y_train)
# joblib.dump(voting_clf, "./models/voting_classifier_model.joblib")

# # Evaluation on Validation Set
# y_pred_voting = voting_clf.predict(X_val)
# print("Voting Classifier Accuracy:", accuracy_score(y_val, y_pred_voting))
# print("Voting Classifier Classification Report:")
# print(classification_report(y_val, y_pred_voting))
