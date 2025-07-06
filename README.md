# **Insurance Loss Modeling – End-to-End Machine Learning Pipeline**

This repository implements a full pipeline for modeling **insurance claim occurrence** and **claim severity**, integrating domain-specific feature engineering with advanced machine learning techniques. The solution is tailored to handle **zero-inflation**, **extreme right-skew**, and **regulatory fairness** in personal insurance.

---

## **Project Scope**

The analysis targets two key insurance outcomes:

- **Claim Status (CS)**: Binary classification (0 = No Claim, 1 = Claim)
- **Loss Severity**:
  - **Loss Cost (LC)** = X.15 / X.16
  - **Historically Adjusted Loss Cost (HALC)** = LC × X.18

These targets pose unique modeling challenges:
- ~89% of policies have **zero claims**
- Loss values exhibit **heavy-tailed distributions**
- Risk of **overpricing low-risk policyholders** if not modeled carefully

---

## **Notebooks**

> `01_preprocessing_and_EDA.ipynb` – Feature engineering, EDA, target creation  
> `02_claim_status_prediction.ipynb` – Classification modeling for CS  
> `03_loss_cost_modeling.ipynb` – Regression modeling for LC and HALC

---

## **1. Data Preparation & Feature Engineering**

### **Domain-Driven Features**
- **Driver Risk Score**: Age < 25, low experience, recent cancellations  
- **Vehicle Risk Score**: Horsepower > 150, old/new age extremes  
- **Customer Loyalty Score**: Tenure, policy mix, products held  
- **Power-to-Weight Ratio** and **Premium-to-Value Ratio** for performance and pricing risk  

### **Preprocessing Techniques**
- Actuarial bucketing for driver and vehicle variables  
- One-hot encoding for categorical features  
- Leakage prevention via temporal variable control  
- Log-transformations for skewed loss targets  

---

## **2. Claim Status Classification (CS)**

### **Final Model: XGBoost Classifier**

#### **Why XGBoost?**
- Highest **ROC-AUC** (0.844) among all classifiers  
- Handles **class imbalance** via `scale_pos_weight`  
- Offers robust interpretability through **SHAP values**  
- Exhibits excellent **calibration and generalization**

#### **Performance Summary**

| Metric              | Value   |
|---------------------|---------|
| **ROC-AUC**         | **0.844** |
| Accuracy            | 77%     |
| Precision (Class 1) | 0.81    |
| Recall (Class 1)    | 0.84    |
| F1 Score (Class 1)  | 0.82    |

#### **Methods Used**
- **Bayesian Optimization (Optuna)** for hyperparameter tuning  
- **5-fold Stratified Cross-Validation**  
- SHAP-based feature selection  
- Benchmarked against: Logistic Regression, LightGBM, Random Forest, Neural Nets  

---

## **3. Claim Severity Regression (LC / HALC)**

### **Final Model: XGBoost Regressor**

#### **Why XGBoost?**
- More stable than LightGBM Three-Part, which overpredicted small claims  
- Lower **mean absolute error** across both LC and HALC  
- Better generalization on **long-tail, high-loss claims**  
- Easier to audit and deploy in production settings

#### **Loss Cost (LC)**

| Metric  | Value     |
|---------|-----------|
| **MSE** | **673,949** |
| MAE     | 114       |
| RMSE    | 820.9     |

#### **Historically Adjusted LC (HALC)**

| Metric  | Value       |
|---------|-------------|
| **MSE** | **1,982,159** |
| MAE     | 197.4       |
| RMSE    | 1,407.9     |

#### **Methods Used**
- **Tweedie regression-aware tuning** for zero-inflation  
- Log-transformed targets  
- Optuna tuning + SHAP feature ranking  
- Evaluated with stratified CV and outlier-resilient metrics  

---

## **4. Machine Learning Concepts & Tools**

| Concept / Tool              | Role in Pipeline                                                  |
|-----------------------------|--------------------------------------------------------------------|
| **Zero-Inflation Modeling** | Log-transform, Tweedie loss, two-part decomposition               |
| **Gradient Boosting**       | XGBoost (final model), LightGBM, CatBoost                         |
| **Neural Networks**         | Feedforward models to benchmark nonlinear patterns                |
| **Hyperparameter Tuning**   | Optuna (Bayesian), GridSearchCV, RandomizedSearchCV               |
| **Class Imbalance**         | SMOTE, class weights, `scale_pos_weight`                          |
| **Model Interpretability**  | SHAP summary plots and importance rankings                        |
| **Evaluation Metrics**      | ROC-AUC, RMSE, MAE, MSE, R², F1-score                              |
| **Actuarial Segmentation**  | Bucketed age, experience, power, vehicle maturity                 |

---

## **5. Visual Diagnostics**

- **ROC Curves**: Classifier performance across thresholds  
- **Prediction Scatterplots**: Accuracy and residual spread across loss ranges  
- **Residual Analysis**: Highlighted over/under-prediction regions  
- **SHAP Interpretability**: Risk scores, loyalty, power ratios among top drivers  

---

## **Conclusion**

This repository delivers a **production-ready ML pipeline** for modeling claim frequency and severity:

- **XGBoost Classifier** for CS: top classification performance and interpretability  
- **XGBoost Regressor** for LC & HALC: best generalization and fairness across all severity levels  

The solution balances **actuarial domain insight**, **ML rigor**, and **deployment readiness** — suitable for use in real-world insurance pricing systems.

