import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from typing import Tuple, Dict, Any


def split_train_val(df: pd.DataFrame, target_col: str, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Розділити датафрейм на навчальну та валідаційну вибірки.

    Аргументи:
        df (pd.DataFrame): Початковий датафрейм.
        target_col (str): Назва цільової колонки для стратифікації.
        test_size (float): Частка вибірки для валідації.
        random_state (int): Фіксоване зерно для відтворюваності.

    Повертає:
        Tuple[pd.DataFrame, pd.DataFrame]: Навчальний і валідаційний датафрейми.
    """
    train_df, val_df = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df[target_col])
    return train_df, val_df


def separate_inputs_targets(df: pd.DataFrame, input_cols: list, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Відокремити ознаки та ціль із датафрейму.

    Аргументи:
        df (pd.DataFrame): Датафрейм.
        input_cols (list): Список колонок ознак.
        target_col (str): Назва цільової колонки.

    Повертає:
        Tuple[pd.DataFrame, pd.Series]: Ознаки (DataFrame) та ціль (Series).
    """
    inputs = df[input_cols].copy()
    targets = df[target_col].copy()
    return inputs, targets


def fit_scaler(train_inputs: pd.DataFrame, numeric_cols: list) -> MinMaxScaler:
    """
    Навчити MinMaxScaler на навчальних даних.

    Аргументи:
        train_inputs (pd.DataFrame): Навчальні ознаки.
        numeric_cols (list): Список числових колонок.

    Повертає:
        MinMaxScaler: Навчений масштабувач.
    """
    scaler = MinMaxScaler().fit(train_inputs[numeric_cols])
    return scaler


def scale_numeric_features(scaler: MinMaxScaler, inputs: pd.DataFrame, numeric_cols: list) -> None:
    """
    Масштабувати числові ознаки за допомогою переданого масштабувача.

    Аргументи:
        scaler (MinMaxScaler): Навчений масштабувач.
        inputs (pd.DataFrame): Ознаки для масштабування.
        numeric_cols (list): Список числових колонок.
    """
    inputs[numeric_cols] = scaler.transform(inputs[numeric_cols])


def fit_encoder(train_inputs: pd.DataFrame, categorical_cols: list) -> OneHotEncoder:
    """
    Навчити OneHotEncoder на категоріальних колонках.

    Аргументи:
        train_inputs (pd.DataFrame): Навчальні ознаки.
        categorical_cols (list): Список категоріальних колонок.

    Повертає:
        OneHotEncoder: Навчений енкодер.
    """
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(train_inputs[categorical_cols])
    return encoder


def encode_categorical_features(encoder: OneHotEncoder, inputs: pd.DataFrame, categorical_cols: list) -> pd.DataFrame:
    """
    Закодувати категоріальні ознаки за допомогою енкодера.

    Аргументи:
        encoder (OneHotEncoder): Навчений енкодер.
        inputs (pd.DataFrame): Ознаки для кодування.
        categorical_cols (list): Список категоріальних колонок.

    Повертає:
        pd.DataFrame: Датафрейм із закодованими категоріальними ознаками.
    """
    encoded = encoder.transform(inputs[categorical_cols])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols), index=inputs.index)
    inputs = inputs.drop(columns=categorical_cols)
    return pd.concat([inputs, encoded_df], axis=1)


def preprocess_data(raw_df: pd.DataFrame, scale_numeric: bool = True) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, list, MinMaxScaler, OneHotEncoder]:
    """
    Попередньо обробити сирий датафрейм.

    Аргументи:
        raw_df (pd.DataFrame): Сирий датафрейм.
        scale_numeric (bool): Чи масштабувати числові ознаки.

    Повертає:
        Tuple:
            - Навчальні ознаки (DataFrame)
            - Навчальні цілі (Series)
            - Валідаційні ознаки (DataFrame)
            - Валідаційні цілі (Series)
            - Список вхідних колонок
            - Навчений масштабувач
            - Навчений енкодер
    """
    # Розбити дані на навчальну та валідаційну вибірки
    train_df, val_df = split_train_val(raw_df, 'Exited')

    # Визначити список вхідних колонок
    input_cols = list(train_df.columns)[1:-1]
    input_cols.remove('Surname')
    target_col = 'Exited'

    # Відокремити ознаки та ціль
    train_inputs, train_targets = separate_inputs_targets(train_df, input_cols, target_col)
    val_inputs, val_targets = separate_inputs_targets(val_df, input_cols, target_col)

    # Знайти числові та категоріальні колонки
    numeric_cols = train_inputs.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = train_inputs.select_dtypes(include='object').columns.tolist()

    # Масштабування числових ознак
    scaler = None
    if scale_numeric:
        scaler = fit_scaler(train_inputs, numeric_cols)
        scale_numeric_features(scaler, train_inputs, numeric_cols)
        scale_numeric_features(scaler, val_inputs, numeric_cols)

    # Кодування категоріальних ознак
    encoder = fit_encoder(train_inputs, categorical_cols)
    train_inputs = encode_categorical_features(encoder, train_inputs, categorical_cols)
    val_inputs = encode_categorical_features(encoder, val_inputs, categorical_cols)

    # Повертаємо всі потрібні об'єкти
    return train_inputs, train_targets, val_inputs, val_targets, input_cols, scaler, encoder


def preprocess_new_data(new_df: pd.DataFrame, input_cols: list, scaler: MinMaxScaler, encoder: OneHotEncoder, scale_numeric: bool = True) -> pd.DataFrame:
    """
    Попередньо обробити нові дані з використанням переданого масштабувача та енкодера.

    Аргументи:
        new_df (pd.DataFrame): Новий датафрейм.
        input_cols (list): Список вхідних колонок.
        scaler (MinMaxScaler): Навчений масштабувач.
        encoder (OneHotEncoder): Навчений енкодер.
        scale_numeric (bool): Чи масштабувати числові ознаки.

    Повертає:
        pd.DataFrame: Оброблені ознаки для нових даних.
    """
    inputs = new_df[input_cols].copy()

    # Знайти числові та категоріальні колонки
    numeric_cols = inputs.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = inputs.select_dtypes(include='object').columns.tolist()

    # Масштабування числових ознак
    if scale_numeric:
        scale_numeric_features(scaler, inputs, numeric_cols)

    # Кодування категоріальних ознак
    inputs = encode_categorical_features(encoder, inputs, categorical_cols)

    return inputs


#if __name__ == "__main__":
    # 1. Зчитати CSV
    #df = pd.read_csv("/Users/mariiakostenko/Downloads/ML/Модуль 3. Дерева прийняття рішень/train.csv")

    # 2. Викликати preprocess_data
    #X_train, y_train, X_val, y_val, input_cols, scaler, encoder = preprocess_data(df)

    # 3. Перевірити результат
    #print("X_train shape:", X_train.shape)
    #print("X_val shape:", X_val.shape)
    #print("Приклад рядка:")
    #print(X_train.iloc[0])

    # 4. Зберегти оброблені дані якщо треба
    #X_train.to_csv("X_train_processed.csv", index=False)
    #X_val.to_csv("X_val_processed.csv", index=False)
