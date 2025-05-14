import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Load dataset
df = pd.read_csv('../dataset/EV_IntelliCare_balanced_dataset_600.csv')

# Features and target
X = df.drop('Health_Status', axis=1)
y = df['Health_Status']

# Label encoding
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# XGBoost Classifier
xgb_model = xgb.XGBClassifier(objective='multi:softmax', num_class=3, eval_metric='mlogloss', use_label_encoder=False)

# Hyperparameter tuning (optional but helps)
params = {
    'n_estimators': [100, 150],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.05, 0.1],
}
grid = GridSearchCV(xgb_model, params, cv=3, scoring='accuracy')
grid.fit(X_train, y_train)

# Best model
model = grid.best_estimator_
y_pred = model.predict(X_test)

# Evaluation
print(f"\n✅ Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

# Save model and encoder
joblib.dump(model, '../models/vehicle_health_model.pkl')
joblib.dump(le, '../models/label_encoder.pkl')
print("✅ Model and encoder saved.")
