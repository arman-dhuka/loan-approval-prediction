import pandas as pd

def load_data(path):
    return pd.read_csv(path)


def drop_columns(df):
    return df.drop("Loan_ID", axis=1)


def handle_missing_values(df):
    # categorical
    df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
    df['Married'].fillna(df['Married'].mode()[0], inplace=True)
    df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
    df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)

    # numerical
    df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)
    df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)

    return df


def fix_data(df):
    df['Dependents'] = df['Dependents'].replace('3+', 3)
    df['Dependents'] = df['Dependents'].astype(int)
    return df


def encode_data(df):
    df['Gender'] = df['Gender'].map({'Male':1, 'Female':0})
    df['Married'] = df['Married'].map({'Yes':1, 'No':0})
    df['Education'] = df['Education'].map({'Graduate':1, 'Not Graduate':0})
    df['Self_Employed'] = df['Self_Employed'].map({'Yes':1, 'No':0})
    df['Loan_Status'] = df['Loan_Status'].map({'Y':1, 'N':0})

    df = pd.get_dummies(df, columns=['Property_Area'], drop_first=True)

    return df


def save_data(df, path):
    df.to_csv(path, index=False)


def preprocess_pipeline(input_path, output_path):
    df = load_data(input_path)
    df = drop_columns(df)
    df = handle_missing_values(df)
    df = fix_data(df)
    df = encode_data(df)
    save_data(df, output_path)


if __name__ == "__main__":
    preprocess_pipeline(
        "../data/raw/loan_data.csv",
        "../data/processed/clean_data.csv"
    )