import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# =====================
# Load dataset
# =====================
df = pd.read_csv("diabetes.csv")
print(df.head())

# =====================
# Data Cleaning
# =====================
df = df.drop(columns=['SkinThickness'])

cols_with_zero = ['Glucose', 'BloodPressure', 'Insulin', 'BMI']
df[cols_with_zero] = df[cols_with_zero].replace(0, np.nan)

# =====================
# Features & Target
# =====================
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# =====================
# Train-test split
# =====================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =====================
# Column selection
# =====================
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns

# =====================
# Preprocessing
# =====================
num_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_pipeline, numeric_features)
    ]
)

# =====================
# Model pipeline
# =====================
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# =====================
# Hyperparameter tuning
# =====================
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5],
    'classifier__min_samples_leaf': [1, 2]
}

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best CV Score:", grid_search.best_score_)

# =====================
# Best model
# =====================
best_model = grid_search.best_estimator_

# =====================
# Evaluation
# =====================
y_pred = best_model.predict(X_test)

print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# =====================
# Save model (IMPORTANT)
# =====================
with open("random_forest_model.pkl", "wb") as file:
    pickle.dump(best_model, file)

print("✅ Model saved as random_forest_model.pkl")