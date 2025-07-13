import numpy as np
import pandas as pd
from typing import Tuple, Dict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load data from a CSV file.

    Args:
        filepath: Path to the CSV file.

    Returns:
        DataFrame containing the loaded data.
    """
    df = pd.read_csv(filepath)
    print(df.head())
    print(df.info())
    return df


def split_data(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into train/validation sets with stratification.

    Args:
        df: The full dataframe.
        target_col: Name of the target column.
        test_size: Fraction of data for validation.
        random_state: Random seed.

    Returns:
        Tuple of train inputs, validation inputs, train targets, validation targets.
    """
    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[target_col]
    )
    input_cols = list(df.columns)[1:-1]  # skip id and target

    train_X = train_df[input_cols].copy()
    train_y = train_df[target_col]
    val_X = val_df[input_cols].copy()
    val_y = val_df[target_col]

    return train_X, val_X, train_y, val_y


def preprocess_data(
    train_X: pd.DataFrame,
    val_X: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Scale numeric columns and one-hot encode categorical columns.

    Args:
        train_X: Training features.
        val_X: Validation features.

    Returns:
        Tuple of preprocessed train and validation features.
    """
    numeric_cols = train_X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = train_X.select_dtypes(include='object').columns.tolist()

    # Scale numeric
    scaler = MinMaxScaler()
    train_X[numeric_cols] = scaler.fit_transform(train_X[numeric_cols])
    val_X[numeric_cols] = scaler.transform(val_X[numeric_cols])

    # One-hot encode categorical
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder.fit(train_X[categorical_cols])

    encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
    train_encoded = encoder.transform(train_X[categorical_cols])
    val_encoded = encoder.transform(val_X[categorical_cols])

    train_X = pd.concat(
        [train_X.drop(columns=categorical_cols).reset_index(drop=True),
         pd.DataFrame(train_encoded, columns=encoded_cols)],
        axis=1
    )
    val_X = pd.concat(
        [val_X.drop(columns=categorical_cols).reset_index(drop=True),
         pd.DataFrame(val_encoded, columns=encoded_cols)],
        axis=1
    )

    return train_X, val_X


def train_model(
    X: pd.DataFrame,
    y: pd.Series
) -> LogisticRegression:
    """
    Train a logistic regression model.

    Args:
        X: Training features.
        y: Training labels.

    Returns:
        Trained LogisticRegression model.
    """
    model = LogisticRegression(solver='liblinear')
    model.fit(X, y)
    return model


def evaluate_model(
    model: LogisticRegression,
    X: pd.DataFrame,
    y_true: pd.Series,
    dataset_name: str
) -> None:
    """
    Print evaluation metrics and plot ROC curve.

    Args:
        model: Trained classifier.
        X: Features.
        y_true: True labels.
        dataset_name: Label for the dataset (e.g., 'Train').
    """
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    auc = roc_auc_score(y_true, y_proba)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print(f"{dataset_name} AUC: {auc:.4f}")
    print(f"{dataset_name} F1 Score: {f1:.4f}")
    print(f"{dataset_name} Confusion Matrix:\n{cm}")

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f'{dataset_name} ROC Curve (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{dataset_name} ROC Curve')
    plt.legend()
    plt.show()


def run_pipeline(filepath: str = '/Users/mariiakostenko/Downloads/ML/Модуль 3. Дерева прийняття рішень/train.csv') -> Dict[str, pd.DataFrame]:
    """
    Main pipeline function to run data loading, preprocessing, training, and evaluation.

    Args:
        filepath: Path to the training CSV.

    Returns:
        Dictionary with train and validation features and targets.
    """
    raw_df = load_data(filepath)

    target_col = 'Exited'
    train_X, val_X, train_y, val_y = split_data(raw_df, target_col)

    train_X, val_X = preprocess_data(train_X, val_X)

    model = train_model(train_X, train_y)

    evaluate_model(model, train_X, train_y, 'Train')
    evaluate_model(model, val_X, val_y, 'Validation')

    return {
        'train_X': train_X,
        'train_Y': train_y,
        'val_X': val_X,
        'val_Y': val_y
    }
