import numpy as np
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import joblib
import os

def train_models(X, y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rf_aucs, xgb_aucs, cat_aucs = [], [], []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        pos_ratio = y_tr.value_counts()[0] / y_tr.value_counts()[1]

        # --- Random Forest ---
        rf = RandomForestClassifier(
            n_estimators=500,
            max_depth=16,
            min_samples_split=4,
            class_weight='balanced',
            n_jobs=-1,
            random_state=42
        )
        rf.fit(X_tr, y_tr)
        rf_probs = rf.predict_proba(X_val)[:, 1]
        rf_auc = roc_auc_score(y_val, rf_probs)
        rf_aucs.append(rf_auc)

        # --- XGBoost ---
        xgb_model = xgb.XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='auc',
            scale_pos_weight=pos_ratio,
            tree_method='hist',
            random_state=42
        )
        xgb_model.fit(X_tr, y_tr)
        xgb_probs = xgb_model.predict_proba(X_val)[:, 1]
        xgb_auc = roc_auc_score(y_val, xgb_probs)
        xgb_aucs.append(xgb_auc)

        # --- CatBoost ---
        cat = CatBoostClassifier(
            iterations=700,
            depth=8,
            learning_rate=0.05,
            loss_function='Logloss',
            eval_metric='AUC',
            class_weights=[1.0, pos_ratio],
            verbose=False,
            random_seed=42
        )
        cat.fit(X_tr, y_tr)
        cat_probs = cat.predict_proba(X_val)[:, 1]
        cat_auc = roc_auc_score(y_val, cat_probs)
        cat_aucs.append(cat_auc)

        print(f"Fold {fold}: RF AUC={rf_auc:.4f}, XGB AUC={xgb_auc:.4f}, CAT AUC={cat_auc:.4f}")

    return rf_aucs, xgb_aucs, cat_aucs


def save_best_model(X, y, rf_aucs, xgb_aucs, cat_aucs):
    mean_scores = {
        'RandomForest': np.mean(rf_aucs),
        'XGBoost': np.mean(xgb_aucs),
        'CatBoost': np.mean(cat_aucs)
    }
    best_name = max(mean_scores, key=mean_scores.get)
    print("\nBest model by mean CV AUC:", best_name, " -> ", mean_scores[best_name])

    pos_ratio_full = y.value_counts()[0] / y.value_counts()[1]

    if best_name == 'RandomForest':
        best_model = RandomForestClassifier(
            n_estimators=500,
            max_depth=16,
            min_samples_split=4,
            class_weight='balanced',
            n_jobs=-1,
            random_state=42
        )
    elif best_name == 'XGBoost':
        best_model = xgb.XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='auc',
            scale_pos_weight=pos_ratio_full,
            tree_method='hist',
            random_state=42
        )
    else:
        best_model = CatBoostClassifier(
            iterations=700,
            depth=8,
            learning_rate=0.05,
            loss_function='Logloss',
            eval_metric='AUC',
            class_weights=[1.0, pos_ratio_full],
            verbose=False,
            random_seed=42
        )

    best_model.fit(X, y)
    os.makedirs("models", exist_ok=True)
    model_filename = f"best_model_{best_name}.pkl"
    model_path = os.path.join("models", model_filename)
    joblib.dump(best_model, model_path)
    print(f"\nSaved best model to: {model_path}")

