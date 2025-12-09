
import pandas as pd

def add_interactions(df: pd.DataFrame) -> pd.DataFrame:
    eps = 1e-6
    
    if {'max_power_value', 'gross_weight'}.issubset(df.columns):
        df['power_to_weight'] = df['max_power_value'] / (df['gross_weight'] + eps)
    if {'max_torque_value', 'gross_weight'}.issubset(df.columns):
        df['torque_to_weight'] = df['max_torque_value'] / (df['gross_weight'] + eps)
    if {'age_of_car', 'policy_tenure'}.issubset(df.columns):
        df['car_age_ratio'] = df['age_of_car'] / (df['policy_tenure'] + 1.0)
    if {'displacement', 'max_power_value'}.issubset(df.columns):
        df['engine_efficiency'] = df['displacement'] / (df['max_power_value'] + eps)
    
    return df
