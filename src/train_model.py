import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib


def load_data(path):
    return pd.read_csv(path)


def split_data(df):
    X = df.drop("Loan_Status", axis=1)
    y = df["Loan_Status"]

    return train_test_split(X, y, test_size=0.2, random_state=42)


def train_model(X_train, y_train):
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ])

    pipeline.fit(X_train, y_train)
    return pipeline


def save_model(model, path):
    # Safety check: only save if it's a trained model, not predictions
    if hasattr(model, "predict"):
        joblib.dump(model, path)
        print(f"✅ Model saved to {path}")
        print(f"   Type: {type(model)}")
    else:
        raise ValueError(
            f"❌ Cannot save: object is {type(model)}, not a trained model. "
            "Make sure you're passing the model, not predictions."
        )


if __name__ == "__main__":
    df = load_data("../data/processed/clean_data.csv")

    X_train, X_test, y_train, y_test = split_data(df)

    model = train_model(X_train, y_train)

    # Evaluate before saving
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"📊 Test Accuracy: {acc:.4f}")

    # Save the MODEL (not y_pred!)
    save_model(model, "../models/loan_model.pkl")