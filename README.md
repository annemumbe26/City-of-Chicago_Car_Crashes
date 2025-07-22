# Predicting Primary Contributory Causes of Car Crashes in Chicago

## Project Overview
This project aims to develop a machine learning model to predict the **primary contributory cause** of car accidents in Chicago. By analyzing historical crash data, the goal is to provide actionable insights to the City of Chicago's Traffic Management Department, enabling more targeted interventions to improve road safety and reduce incidents.

## Business Problem
The City of Chicago seeks to move beyond reactive measures to **proactive prevention** of car accidents. Identifying the primary causes of crashes is crucial for efficient resource allocation, designing effective public safety campaigns, and implementing strategic infrastructure improvements to reduce accidents, injuries, and fatalities.

## Data Sources
The data for this project is sourced from the [Chicago Data Portal](https://data.cityofchicago.org), utilizing two datasets:

- **Traffic_Crashes_Crashes.csv**: Contains crash-level details (e.g., date, time, location, environmental conditions, primary cause).
- **Traffic_Crashes_People.csv**: Contains person-level details involved in crashes (e.g., age, sex, injury classification, BAC results).

## Methodology

### Data Loading & Initial Cleaning
- Datasets loaded using `pd.read_csv` with `nrows=100000` to manage memory for large files.
- `on_bad_lines='skip'` and `encoding='latin1'` used to handle parsing errors.
- Columns with high missingness (>80%) and redundant identifiers were removed.

### Data Merging & Feature Engineering
- Person-level data aggregated (e.g., `NUM_PEOPLE_INVOLVED`, `PEOPLE_INJURY_` counts, `CRASH_HAS_ALCOHOL_TEST`, `CRASH_HAS_DRIVER`) and merged with crash-level data using `CRASH_RECORD_ID` to create `combined_df`.
- Temporal features extracted from `CRASH_DATE`: `CRASH_YEAR`, `CRASH_MONTH`, `CRASH_DAY_OF_WEEK`, `CRASH_HOUR`, `IS_WEEKEND`, `IS_RUSH_HOUR`.
- Rare `PRIM_CONTRIBUTORY_CAUSE` categories (<0.5% of records) grouped into an `'OTHER'` class.

### Preprocessing Pipeline
- Data split into training (80%) and testing (20%) using `train_test_split` with `stratify=y`.
- `ColumnTransformer` used for feature preprocessing:
  - **Numerical**: Imputed with median, then scaled (`StandardScaler`).
  - **Categorical**: Imputed with most frequent, then one-hot encoded (`OneHotEncoder`).

### Model Training & Imbalance Handling
- Baseline models: Logistic Regression, Decision Tree, Random Forest with `class_weight='balanced'` and integrated with **SMOTE** using `imblearn.pipeline.Pipeline`.
- Advanced model: **LightGBM Classifier** integrated with SMOTE.

### Hyperparameter Tuning & Evaluation
- `GridSearchCV` used to tune LightGBM hyperparameters (`n_estimators`, `learning_rate`, `num_leaves`, `max_depth`) with `cv=3` and `scoring='f1_weighted'`.

**Evaluation Metrics:**
- **Accuracy**: Overall prediction correctness.
- **Classification Report**: Precision, recall, F1-score per class.
- **Confusion Matrix**: Visual breakdown of predictions.
- **Feature Importance Plot**: Highlights most influential features.

## Key Findings & Results

### LightGBM as Best Performer
- Outperformed Logistic Regression and Decision Tree in both accuracy and F1-weighted score.
- **Best LightGBM Test Accuracy**: ~0.4138
- **Best F1-weighted Score (Cross-Validated)**: ~0.3779

### Random Forest Overfitting
- Severe overfitting observed (train accuracy ~1.0, much lower test accuracy).

### Class Imbalance
- Despite `class_weight='balanced'` and **SMOTE**, many minority classes suffered from low recall and F1-scores.
- Prevalent causes like `"UNABLE TO DETERMINE"` and `"FAILING TO REDUCE SPEED TO AVOID CRASH"` had better predictive accuracy.

### Most Important Features (LightGBM)
- `CRASH_HOUR` and `BEAT_OF_OCCURRENCE` were most influential.
- Other key features:
  - `CRASH_DAY_OF_WEEK`, `CRASH_MONTH`, `CRASH_YEAR`
  - `POSTED_SPEED_LIMIT`, `IS_RUSH_HOUR`, `NUM_UNITS`
  - Various `FIRST_CRASH_TYPE_` indicators

## Limitations
- **Data Sampling**: Only first 100,000 rows used due to memory constraints, limiting generalizability.
- **Class Imbalance**: Despite mitigation efforts, predicting rare causes remains a major challenge.

## Recommendations & Future Work

### Improve Imbalance Handling
- Explore **SMOTE-Tomek**, **SMOTE-ENN**, or **cost-sensitive learning**.

### Advanced Hyperparameter Optimization
- Use `RandomizedSearchCV`, **Optuna**, or **Hyperopt** for better tuning.

### Enhanced Feature Engineering
- Combine features (e.g., `WEATHER_CONDITION` × `LIGHTING_CONDITION`).
- Include external sources (e.g., traffic flow, road work schedules).
- Leverage **geospatial features** like crash hotspots if `LATITUDE/LONGITUDE` are reintroduced.

### Error Analysis
- Qualitative review of misclassified samples to discover overlooked patterns.

### Model Ensembling
- Try **stacking or blending** LightGBM with models like **XGBoost** for better performance.

---

This project offers a strong foundation for identifying car crash causes in Chicago and outlines clear paths for model improvement and deeper analysis.
