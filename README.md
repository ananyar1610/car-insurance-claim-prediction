#  Car Insurance Claim Prediction

A complete end-to-end **machine learning** and **Streamlit** project that predicts whether a car insurance policy will result in a **claim** in the next policy period, based on customer, vehicle, and policy features.  

This repository is structured as a **deployment-ready ML project** with:

- Clean preprocessing & feature engineering
- Model training with cross-validation (Random Forest, XGBoost, CatBoost)
- Saved production model (`.pkl`)
- Interactive **Streamlit web app** for real-time prediction

---

##  Live Demo

> **Streamlit App:** _Add your deployed URL here_  
> For example:  
> `https://your-app-name-your-username.streamlit.app`

---

##  Table of Contents

- [Problem Statement](#-problem-statement)
- [Business Use Cases](#-business-use-cases)
- [Skills & Tech Stack](#-skills--tech-stack)
- [Dataset](#-dataset)
- [Feature Description](#-feature-description)
- [Project Structure](#-project-structure)
- [Modeling Approach](#-modeling-approach)
- [Streamlit Application](#-streamlit-application)
- [How to Run Locally](#-how-to-run-locally)
- [How to Train / Retrain the Model](#-how-to-train--retrain-the-model)
- [Deployment on Streamlit Cloud](#-deployment-on-streamlit-cloud)
- [Reproducibility](#-reproducibility)
- [Limitations & Future Work](#-limitations--future-work)
- [License & Data Usage](#-license--data-usage)
- [Acknowledgements](#-acknowledgements)

---

##  Problem Statement

**Car Insurance Claim Prediction**  

> Build a classification model that predicts whether a customer will make a **car insurance claim** in the next policy period using demographic, vehicle, and policy-related features. :contentReference[oaicite:0]{index=0}  

The model output (`is_claim`) is **binary**:

- `1` → The policyholder **filed** a claim  
- `0` → The policyholder **did not** file a claim  

---

##  Business Use Cases

This project is aligned with typical **insurance / risk analytics** use cases: :contentReference[oaicite:1]{index=1}  

1. **Fraud / Risk Prevention**  
   Identify high-risk customers early and adjust underwriting rules or perform extra checks.

2. **Pricing Optimization**  
   Use predicted claim probability to support **risk-based pricing** and fair premium calculation.

3. **Customer Targeting**  
   Design retention and marketing campaigns for **low-risk customers**.

4. **Operational Efficiency**  
   Forecast claim volumes to help claim departments **plan capacity and resources**.

---

##  Skills & Tech Stack

**Skills demonstrated**

- Exploratory Data Analysis (EDA) & Visualization  
- Data Preprocessing (missing values, encoding, scaling)  
- Feature Engineering & Selection  
- Supervised Classification (tree-based models)  
- Cross-validation & hyperparameter tuning  
- Model evaluation (ROC-AUC, F1, Precision/Recall)  
- Streamlit app design & deployment  
- Git-based version control & PEP8-compliant code :contentReference[oaicite:2]{index=2}  

**Tech stack**

- **Language:** Python  
- **Core libs:** `pandas`, `numpy`, `scikit-learn`  
- **Models:** `RandomForestClassifier`, `XGBoost`, `CatBoost`  
- **App:** `streamlit`  
- **Persistence:** `joblib`  
- **Version control:** `git`

---

##  Dataset

- **Source (Provided by GUVI):**  
  `https://drive.google.com/file/d/1RP5vqMcI9SIFW3LsdacdHoTtrAgylC8l/view` :contentReference[oaicite:3]{index=3}  

>  **Important:**  
> The dataset is **not redistributed** in this repository.  
> To run the project end-to-end, please **download the dataset** from the above link and place the files in the appropriate data folder (see below).

Typical files:

- `train.csv` – Training data with target `is_claim`
- `test.csv` – Test / evaluation data without labels (for prediction / submission)

---

##  Feature Description

Below is a summary of key features from the project spec. :contentReference[oaicite:4]{index=4}  

### Policy & Customer Features

- `policy_id` – Unique identifier of the policy  
- `policy_tenure` – Duration of the policy  
- `age_of_car` – Normalized age of the car  
- `age_of_policyholder` – Normalized age of the policyholder  
- `area_cluster` – Encoded area / region of the customer  
- `population_density` – Population density of the policyholder’s city  

### Vehicle Identification & Category

- `make` – Encoded car manufacturer  
- `segment` – Segment/category of the car (A, B1, B2, C1, C2)  
- `model` – Encoded model name of the car  
- `fuel_type` – Fuel type (Petrol, Diesel, CNG, etc.)  
- `engine_type` – Type of engine used  

### Engine Performance & Specs

- `max_power` – Maximum power (bhp@rpm)  
- `max_torque` – Maximum torque (Nm@rpm)  
- `displacement` – Engine displacement in cc  
- `cylinder` – Number of cylinders  

During preprocessing, `max_power` and `max_torque` are parsed into numeric components:

- `max_power_value`, `max_power_rpm`  
- `max_torque_value`, `max_torque_rpm`

### Transmission & Control

- `transmission_type` – Gearbox type (manual/automatic/etc.)  
- `gear_box` – Number of gears  
- `rear_brakes_type` – Type of rear brakes  
- `steering_type` – Type of steering  

### Dimensions & Handling

- `turning_radius` – Minimum turning radius (m)  
- `length`, `width`, `height` – Vehicle dimensions (mm)  
- `gross_weight` – Maximum allowable loaded weight (kg)  

### Safety Features (Booleans)

All of these are stored as **Yes/No** in raw data and converted to 0/1:

- `airbags` – Number of airbags  
- `is_esc` – Electronic Stability Control  
- `is_adjustable_steering` – Adjustable steering  
- `is_tpms` – Tyre Pressure Monitoring System  
- `is_parking_sensors` – Parking sensors  
- `is_parking_camera` – Parking camera  
- `is_front_fog_lights`  
- `is_rear_window_wiper`  
- `is_rear_window_washer`  
- `is_rear_window_defogger`  
- `is_brake_assist`  
- `is_power_door_lock`  
- `is_central_locking`  
- `is_power_steering`  
- `is_driver_seat_height_adjustable`  
- `is_day_night_rear_view_mirror`  
- `is_ecw` – Engine Check Warning  
- `is_speed_alert` – Over-speed alert  
- `ncap_rating` – NCAP safety rating (1–5)  

### Target

- `is_claim` – **Binary target**  
  - `1` → Claim filed  
  - `0` → No claim

##  Modeling Approach (High-Level)

### **Preprocessing**
- Map **Yes/No → 1/0** for boolean flags  
- Extract numeric values from `max_power` and `max_torque` strings  
- Label-encode categorical features (e.g., `fuel_type`, `segment`, `model`, `engine_type`, etc.)  
- Save encoders to **`all_encoders.pkl`** for reuse in the Streamlit app  

---

### **Feature Engineering**
To improve ML performance, several derived features are created:

- `power_to_weight = max_power_value / gross_weight`  
- `torque_to_weight = max_torque_value / gross_weight`  
- `car_age_ratio = age_of_car / (policy_tenure + 1)`  
- `engine_efficiency = displacement / max_power_value`  

---

### **Model Training**
- Use **StratifiedKFold (5 folds)** for balanced evaluation  
- Train multiple models:
  - `RandomForestClassifier`
  - `XGBClassifier`
  - `CatBoostClassifier`  
- Evaluate each model using **mean ROC-AUC** across folds  
- Pick the **best-performing model**  
- Retrain the best model on the **full dataset**  
- Save final model as:  
  - `models/best_model_<ModelName>.pkl`

---

### **Evaluation Metrics**
- **ROC-AUC**  
- **F1-score**  
- **Precision / Recall**  
- **Accuracy**  
- **Confusion Matrix** (during experimentation)

---

##  Streamlit Application

The Streamlit app (`app.py`) contains three main sections:

---

### **1️ Home**
- Project introduction  
- Business context & problem statement  
- Instructions on how to use the app  
- Optional banner images using `st.image()`  

---

### **2️ Data Overview**
- Metric cards:  
  - Number of rows  
  - Number of columns  
  - Claim rate  
- Sample rows from `train.csv`  
- Column descriptions & datatypes  
- Target distribution plot (`is_claim`)  
- Optional:
  - Toggle view between **train** and **test** datasets  

---

### **3️ Prediction**
User provides live input through:

- **Categorical fields** → dropdowns using fitted label encoders  
- **Numeric fields** → `st.number_input`  
- **Boolean fields** → Yes/No mapped to 1/0  
- All feature engineering applied exactly as in training  
- Model generates:
  - Predicted **class** → 0 (No Claim) / 1 (Claim)  
  - Predicted **probability** (optional)

---

##  How to Run Locally

### **1. Clone the Repository**
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

### **2. Create and Activate a Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate      # macOS / Linux
venv\Scripts\activate         # Windows
```

### **3. Install Dependencies**
```
pip install -r requirements.txt
```

### **4. Place Data Files**

Download the dataset from Google Drive and place:

train.csv  → project root (or data/raw/)
test.csv   → same folder as train.csv


Ensure paths inside app.py match your chosen location.

### **5. Run Streamlit App**
```bash
streamlit run app.py
```

Open the generated URL (typically → http://localhost:8501).

---

##  How to Train / Retrain the Model

If you maintain a training script (src/model_training.py):

```bash
python src/model_training.py
```

This will:

-Load and preprocess data

-Perform cross-validation

-Select best model

-Retrain on the full dataset

### **Save:**

```bash
models/best_model_<ModelName>.pkl

models/all_encoders.pkl
```

Ensure your Streamlit app loads these same files.

---

##  Deployment on Streamlit Cloud

- Push the following to a public GitHub repo:

  - app.py

  - requirements.txt

  - models/*.pkl

  - Optional: images/, data/raw/README.md

- Go to Streamlit Community Cloud → https://streamlit.io/

- Click New app, then select:

  - Repo: <your-username>/<repo-name>

  - Branch: main

  - Main file: app.py

- Click Deploy.

  -Your web app will get a public URL you can share.

---

## 🔁 Reproducibility

This project follows ML best practices:

- PEP8 code formatting

- Fixed random_state for deterministic training

- Clear data structure:

- data/raw/ → unmodified raw data

- data/processed/ → ignored in Git (intermediate artifacts)

- Encoders & models saved using joblib

- requirements.txt fully pins dependencies

---

## ⚠ Limitations & Future Work

Model trained on a single dataset; generalization requires additional data

### Potential improvements:

- Advanced regularization / feature selection

- Probability calibration

- Model explainability (e.g., SHAP)

- Handling concept drift (changes in claim behavior over time)

- Containerized deployment (Docker + FastAPI)

---

### 📜 License & Data Usage

#### Code:
Consider using MIT / Apache-2.0 License.

#### Data:
The dataset is not redistributed in this repository.
Users must download the dataset separately from the provided Google Drive link and place it manually before running the pipeline.

Always respect the dataset’s original usage terms.
