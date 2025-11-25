# Motor Insurance Claim Prediction

This project builds a machine learning pipeline to predict whether a motor insurance policy will result in a claim.
The pipeline emphasizes clean preprocessing, reproducibility, and robust model evaluation.

### 1. Problem Statement

Given policy and vehicle information (model, segment, fuel type, safety features, area cluster, etc.),
predict the probability that a customer will make a claim in the next policy period.

- Business Objectives

- Identify high-risk policies

- Improve risk-based pricing

- Support fraud detection

- Reduce losses through better risk management

### 2. Project Structure

A recommended folder layout:
.
├── data/
│   ├── train.csv            # raw training data (never modified)
│   └── test.csv             # raw test data (never modified)
├── src/
│   └── train_models.py      # main ML pipeline script
├── notebooks/
│   └── exploration.ipynb    # EDA, visualizations, notes
├── README.md
└── requirements.txt

Notes

- data/ contains raw CSVs; do NOT overwrite them.

- src/train_models.py performs:

  - Loading data

  - Preprocessing + feature engineering

  - Target encoding + imputation

  - Hyperparameter tuning via RandomizedSearchCV

  - Final model selection + evaluation

- notebooks/ is optional for EDA or documentation.

- requirements.txt ensures experiment reproducibility.

### 3. Setup & Installation

1. Clone the repository
```git clone <your-repo-url>.git
cd <your-repo-folder>
```

2. Create & activate a virtual environment

-Linux / macOS:

```python -m venv venv
source venv/bin/activate
```


-Windows (PowerShell):

```python -m venv venv
venv\Scripts\activate
```

3. Install dependencies
`pip install -r requirements.txt`

4. Place data files
data/
├── train.csv
└── test.csv

### 4. How to Run the Pipeline

#### From the project root:

`python src/train_models.py`

The script will:

- Set global seeds (RANDOM_SEED = 42) for reproducibility.

- Load train.csv and test.csv.

- Apply preprocessing:

  - Convert Yes/No boolean features → 1/0

  - Parse max_power and max_torque into numeric fields

  - Target-encode: model, engine_type, segment, area_cluster (OOF = leak-safe)

  - Label-encode: fuel, brakes, transmission, steering

  - Feature engineering (power-to-weight, torque-to-weight, engine_efficiency, car_age_ratio)

  - Add safety_score (sum of safety features)

  - Add missing-value indicator columns

  - Impute with median

- Run RandomizedSearchCV over:

  - RandomForest

  - XGBoost

  - CatBoost

- Select the best model by CV ROC-AUC.

- Evaluate the best model on a hold-out validation set using:

  - Accuracy

  - Precision

  - Recall

  - F1-score

  - ROC-AUC

  - Confusion matrix

- Optionally write predictions to:

`submission_tuned_all_models.csv`

### 5. Model Evaluation
Metrics Used

- Accuracy – overall correctness

- Precision (class 1) – claims correctly flagged

- Recall (class 1) – percentage of true claims identified

- F1-Score – balance between precision & recall

- ROC-AUC – ability to rank policies by risk

- Confusion Matrix – counts of TP, FP, TN, FN

#### Why these metrics?

- High Recall → fewer missed high-risk policies

- High Precision → fewer false investigations

- High AUC → better risk ranking for pricing & fraud detection

### 6. Reproducibility

#### This project ensures reproducibility through:

- Fixed RANDOM_SEED = 42 applied to:

  - NumPy

  - Python random

  - StratifiedKFold

  - RandomForest / XGBoost / CatBoost

  - RandomizedSearchCV

- All transformations implemented in src/train_models.py.

- Environment dependencies listed in requirements.txt.

- Generate/update dependencies:

`pip freeze > requirements.txt`

### 7. Version Control (Git Guidelines)
#### Basic workflow

1. Check changes:

`git status`


2. Stage changes:

`git add src/train_models.py`


3. Commit:

`git commit -m "Improve preprocessing and add RandomizedSearchCV"`


4. Push to GitHub:

`git push origin main`

####Branching (recommended for experiments)
`git checkout -b feature/new-modeling-approach`


Work as usual, then push:

`git push origin feature/new-modeling-approach`

### 8. Limitations & Future Enhancements
Current limitations

- Dataset has weak signal (ROC-AUC ≈ 0.62 even after tuning).

- Only static features; missing:

  - Driver history

  - Past claim patterns

  - External risk indicators (weather, traffic, theft index)

  - Telematics / behavioral data

Potential improvements

  - Add richer, behavioral features

  - Use calibrated prediction (isotonic / Platt scaling)

  - Stacking / blending multiple tuned models

  - Try deep learning for categorical encodings (TabNet, FT-Transformer)

  - SMOTE or focal loss for handling class imbalance

### 9. Contact / Contribution

If you'd like to contribute:

- Fork the repo

- Create a feature branch

- Submit a pull request

**Suggestions and improvements are welcome!**
   
