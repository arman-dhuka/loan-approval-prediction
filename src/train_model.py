import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib


def load_data(path):
    return pd.read_csv(path)


def split_data(df):
    X = df.drop("Loan_Status", axis=1)
    y = df["Loan_Status"]

    return train_test_split(X, y, test_size=0.2, random_state=42)


def train_model(X_train, y_train):
    # pipeline = Pipeline([
    #     ("scaler", StandardScaler()),
    #     ("model", RandomForestClassifier())
    # ])
    log_pipeline = Pipeline([
    ("Scaler" , StandardScaler()),
    ("model" , LogisticRegression(max_iter=1000))
])

    log_pipeline.fit(X_train, y_train)
    return log_pipeline


def save_model(model, path):
    joblib.dump(model, path)


if __name__ == "__main__":
    df = load_data("../data/processed/clean_data.csv")

    X_train, X_test, y_train, y_test = split_data(df)

    model = train_model(X_train, y_train)

    save_model(model, "../models/loan_model.pkl")