import pandas as pd

def feature_engineering(df):
    df = df.copy()

    if "Loan_ID" in df.columns:
        df.drop("Loan_ID", axis=1, inplace=True)

    df["Total_Income"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
    df["Income_Loan_Ratio"] = df["Total_Income"] / df["LoanAmount"]

    return df