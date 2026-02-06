import pandas as pd
import pickle,joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ======================
# Load dataset
# ======================
df = pd.read_csv("utf-8''all_data.csv")   # make sure CSV is in same folder
df.dropna(inplace=True)

# ======================
# Split features & target
# ======================
X = df.drop("stroke", axis=1)
y = df["stroke"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.8, random_state=42
)

# ======================
# Identify columns
# ======================
num_cols = X_train.select_dtypes(include="number").columns
cat_cols = X_train.select_dtypes(exclude="number").columns

# ======================
# Preprocessing
# ======================
preprocessing = ColumnTransformer(
    transformers=[
        ("scaler", StandardScaler(), num_cols),
        ("onehot", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), cat_cols),
    ]
)

# ======================
# Model pipeline
# ======================
model_pipeline = Pipeline(
    steps=[
        ("preprocessing", preprocessing),
        ("model", LogisticRegression(class_weight="balanced"))
    ]
)

# ======================
# Train model
# ======================
model_pipeline.fit(X_train, y_train)

# ======================
# Evaluation
# ======================
y_pred = model_pipeline.predict(X_test)

print("Train Accuracy:", model_pipeline.score(X_train, y_train))
print("Test Accuracy:", model_pipeline.score(X_test, y_test))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ======================
# Save model using pickle
# ======================
with open("stroke_model.pkl", "wb") as file:
    pickle.dump(model_pipeline, file)

print("\nModel saved as stroke_model.pkl")