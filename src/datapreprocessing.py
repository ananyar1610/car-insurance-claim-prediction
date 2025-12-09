
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib

def load_data(train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Load train and test data
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test

def clean_data(df_train: pd.DataFrame, df_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Convert Boolean columns
    bool_cols = [
        'is_esc','is_adjustable_steering','is_tpms','is_parking_sensors','is_parking_camera',
        'is_front_fog_lights','is_rear_window_wiper','is_rear_window_washer','is_rear_window_defogger','is_brake_assist',
        'is_power_door_locks','is_central_locking','is_power_steering','is_driver_seat_height_adjustable',
        'is_day_night_rear_view_mirror','is_ecw','is_speed_alert'
    ]
    for col in bool_cols:
        df_train[col] = df_train[col].map({'Yes': 1, 'No': 0})
        df_test[col] = df_test[col].map({'Yes': 1, 'No': 0})
    
    df_train = df_train.drop('policy_id', axis=1)
    
    return df_train, df_test

def encode_categorical(df_train: pd.DataFrame, df_test: pd.DataFrame, categorical_columns: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        df_train[col] = le.fit_transform(df_train[col])
        df_test[col] = le.transform(df_test[col])
        encoders[col] = le
    return df_train, df_test, encoders

def save_encoders(encoders: dict, filepath: str) -> None:
    joblib.dump(encoders, filepath)
