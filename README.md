# Customer Churn Prediction & Revenue Risk Dashboard

An end-to-end data science project that predicts which telecom customers are likely to churn, explains *why* they're at risk, and quantifies the resulting revenue impact for the business — from raw data to a 3-page Power BI dashboard.

## Problem

Subscription and telecom businesses lose recurring revenue every time a customer leaves. Acquiring a new customer is far more expensive than retaining an existing one, so identifying at-risk customers *before* they churn — and understanding what's driving that risk — lets a retention team act early instead of reacting after the fact.

## Dataset

**IBM Telco Customer Churn** — 7,043 customers, 21 features (demographics, account info, subscribed services, billing details), loaded directly from IBM's public GitHub repository.

- Churn rate: **26.5%** (1,869 churned / 5,174 retained) — a meaningfully imbalanced target
- Known data quality issue: `TotalCharges` loads as text due to 11 blank values (new customers with 0 tenure)

## Approach

The project follows a full ML lifecycle, split into four notebooks:

### 1. Exploratory Data Analysis (`01_eda.ipynb`)
- Checked structure, nulls, duplicates, and data types
- Confirmed the churn class imbalance (26.5%)
- Visualized churn against contract type, tenure, monthly charges, payment method, and internet service type
- Correlation heatmap on numeric features

### 2. Feature Engineering (`02_Feature_engineering.ipynb`)
- Converted `TotalCharges` to numeric and filled 11 missing values with the median
- Dropped `customerID` (no predictive value)
- Created `tenure_bucket` (0-12, 13-24, 25-48, 48+ months) to capture the new-vs-long-term customer pattern seen in EDA
- Created `services_count` — total number of add-on services (Online Security, Tech Support, Streaming TV, etc.) per customer
- Saved a cleaned dataset for modeling

### 3. ML Pipeline (`03_modeling_pipeline.ipynb`)
- Built a single `scikit-learn` Pipeline combining:
  - `ColumnTransformer` — median imputation + scaling for numeric features, most-frequent imputation + one-hot encoding for categorical features
  - **SMOTE** (via `imblearn`) to correct the churn class imbalance during training
  - A classifier
- Stratified 80/20 train/test split to preserve the churn ratio in both sets
- Compared three models by ROC-AUC:

  | Model | ROC-AUC |
  |---|---|
  | Logistic Regression | 0.839 |
  | Random Forest | 0.820 |
  | XGBoost | 0.818 |

- Tuned Random Forest with `GridSearchCV` (3-fold, scored on ROC-AUC) — best cross-validated score: **0.844**
- Saved the final trained pipeline and train/test splits for evaluation

### 4. Model Evaluation (`04_model_evalution.ipynb`)
- Generated predictions and churn probabilities on the held-out test set
- Evaluated with precision, recall, F1, ROC-AUC, and a confusion matrix — not just accuracy, since the data is imbalanced
- Explained predictions with SHAP feature importance
- Framed results in business terms (revenue at risk)
- Exported `churn_predictions.csv`, `confusion_matrix.csv`, and `feature_importance.csv` for the Power BI dashboard

## Key Findings

- Month-to-month contracts, low tenure (<12 months), electronic-check payment, and fiber-optic internet are all associated with meaningfully higher churn
- `Contract` type and `tenure` are the strongest predictors of churn, consistent across both the EDA correlation checks and the model's feature importance

## Model Results (test set, 1,409 customers)

| Metric | No Churn | Churn |
|---|---|---|
| Precision | 0.87 | 0.55 |
| Recall | 0.80 | 0.68 |
| F1-score | 0.84 | 0.61 |

- **Overall accuracy:** 77%
- **ROC-AUC:** 0.838
- The model catches **68% of actual churners** (recall) — the metric that matters most for a retention use case, since missing a churner is costlier than a false alarm

## Business Impact

- **460 customers** in the test set were flagged as at-risk
- At an average monthly charge of **$64.09**, this represents approximately **$29,481/month** in potentially retainable revenue if the retention team acts on the flagged list

## Power BI Dashboard

A 3-page interactive dashboard built from the model's output:

1. **Executive Overview** — KPI cards (Total Customers, Churn Rate %, Predicted Churners, Revenue At Risk), churn-by-segment bar charts (Contract, Payment Method), a Low/Medium/High risk-level donut chart, and a sortable, filtered table of the highest-risk customers
2. **Customer Detail** — full customer table, a tenure-vs-charges scatter plot colored by risk level, and slicers for Contract, Internet Service, Payment Method, and tenure bucket
3. **Model Performance** — ROC-AUC/precision/recall KPI cards, a confusion matrix, and a feature importance chart, for anyone wanting to verify the model itself

## Tech Stack

- **Python:** pandas, seaborn/matplotlib, scikit-learn, imbalanced-learn (SMOTE), XGBoost, SHAP
- **BI:** Power BI (DAX measures and calculated columns)
- **Dataset:** IBM Telco Customer Churn

## How to Run

1. Clone this repository
2. Install dependencies:
   ```
   pip install pandas numpy scikit-learn imbalanced-learn xgboost shap seaborn matplotlib joblib
   ```
3. Run the notebooks in order:
   - `01_eda.ipynb`
   - `02_Feature_engineering.ipynb`
   - `03_modeling_pipeline.ipynb`
   - `04_model_evalution.ipynb`
4. Open `powerbi/telco_customer_churn.pbix` in Power BI Desktop to explore the dashboard

## Project Structure

```
Customer-Churn-Prediction/
│
├── data/
│   └── telco_customer_churn.csv
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling_pipeline.ipynb
│   └── 04_model_evaluation.ipynb
│
├── outputs/
│   ├── churn_pipeline.pkl
│   ├── churn_predictions.csv
│   ├── confusion_matrix.csv
│   └── feature_importance.csv
│
├── powerbi/
│   └── telco_customer_churn.pbix
│
├── README.md
└── requirements.txt
