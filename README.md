# 🛒 E-Commerce Customer Churn Prediction

Complete end-to-end machine learning project for predicting customer churn in e-commerce using the Brazilian E-Commerce (Olist) dataset.

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.2-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0.3-red.svg)](https://xgboost.readthedocs.io/)

## 📊 Project Overview

**Goal**: Predict which customers are likely to churn (stop purchasing) to enable proactive retention campaigns.

**Dataset**: 96,478 customers from Olist (Brazilian e-commerce platform)  
**Churn Rate**: 59.82%  
**Model Performance**: **ROC-AUC 0.90+**  
**Business Impact**: Potential to save **$1.4M+ in revenue**

---

## 🗂️ Project Structure

```
ecommerce-churn-prediction/
│
├── data/
│   ├── raw/                          # Original Olist datasets (8 CSV files)
│   └── processed/                    # Processed features
│       ├── customer_features_selected.csv    # Final dataset (22 features)
│       ├── selected_features.txt             # Feature list
│       └── feature_selection_metadata.json   # Selection stats
│
├── notebooks/
│   ├── 01_eda.ipynb                  # Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb  # RFM Analysis & Feature Creation
│   └── 03_modeling.ipynb             # Model Training & Evaluation
│
├── src/
│   ├── data_processing.py            # Data loading utilities
│   ├── feature_engineering.py        # Feature creation functions
│   └── predict_churn.py              # Inference script (ChurnPredictor class)
│
├── models/
│   ├── churn_prediction_model.pkl    # Trained XGBoost model
│   ├── scaler.pkl                    # StandardScaler for features
│   ├── feature_list.txt              # 22 selected features
│   └── model_metadata.json           # Model info & metrics
│
├── results/
│   ├── target_distribution.png       # Class distribution
│   ├── roc_curves.png                # ROC curve comparison
│   ├── confusion_matrices.png        # Model predictions
│   ├── feature_importance_comparison.png
│   ├── model_comparison.csv          # All model metrics
│   └── business_insights_report.txt  # Actionable recommendations
│
├── requirements.txt                  # Python dependencies
├── README.md                         # This file
└── .gitignore
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/yourusername/ecommerce-churn-prediction.git
cd ecommerce-churn-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Data

Download the [Olist E-Commerce Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) from Kaggle and place CSV files in `data/raw/`.

### 3. Run Notebooks (Training Pipeline)

```bash
jupyter notebook

# Run in order:
# 1. notebooks/01_eda.ipynb
# 2. notebooks/02_feature_engineering.ipynb
# 3. notebooks/03_modeling.ipynb
```

### 4. Make Predictions (Inference)

```python
from src.predict_churn import ChurnPredictor

# Load model
predictor = ChurnPredictor(model_dir='models/')

# Predict for new customers
predictions = predictor.predict(customer_data)

# Get top 100 at-risk customers
top_risk = predictor.get_top_risk_customers(customer_data, top_n=100)
```

**Command-line usage:**
```bash
python src/predict_churn.py \
    --input data/new_customers.csv \
    --output predictions.csv \
    --top-n 100
```

---

## 📈 Methodology

### 1. Exploratory Data Analysis
- **Dataset**: 8 CSV files (orders, customers, products, reviews, etc.)
- **Time range**: 2016-2018
- **Key findings**: 
  - Single-order customers: 97%
  - Average order value: $137 BRL
  - Peak order time: 16:00

### 2. Feature Engineering

**Created 55 features across 9 categories:**

| Category | Features | Examples |
|----------|----------|----------|
| **RFM** | 3 | recency, frequency, monetary |
| **Temporal** | 2 | customer_lifetime_days, avg_days_between_orders |
| **Order Value** | 4 | avg_order_value, min/max/std |
| **Payment** | 4 | payment types (boleto, credit, debit), installments |
| **Review** | 3 | avg/min/max review scores |
| **Product** | 2 | unique_categories, top_category |
| **Delivery** | 2 | avg/max delivery time |
| **Segment** | 8 | RFM segments (Champions, At Risk, Lost, etc.) |
| **Time** | 23 | hour patterns, weekend ratio, time of day |

**Feature Selection**: Reduced to **22 optimized features** (57% reduction)
- Removed high-correlation features (redundant)
- Removed low-importance features
- Retained all top-10 predictive features

### 3. Model Training

**Models Tested:**
1. **Logistic Regression** (Baseline)
   - ROC-AUC: 0.86
   - Fast, interpretable
   
2. **Random Forest**
   - ROC-AUC: 0.89
   - Handles non-linearity
   
3. **XGBoost** ⭐ **BEST**
   - ROC-AUC: **0.91**
   - Precision: **0.79**
   - Recall: **0.82**
   - F1-Score: **0.80**

**Hyperparameter Tuning**: RandomizedSearchCV with 5-fold stratified CV

---

## 🎯 Key Findings

### Top Churn Drivers

1. **Recency** (57% importance) 🔴
   - Most critical factor
   - Customers inactive >180 days → 80%+ churn probability
   
2. **RFM Segments** (40% importance)
   - Champions: Lowest churn (12%)
   - At Risk: Highest churn (85%)
   
3. **Delivery Time** (2% importance)
   - Slow delivery increases churn risk
   - Average: 12 days → target <10 days

4. **Review Scores** (1% importance)
   - Low satisfaction (score <3) → churn
   
### Customer Risk Segmentation

| Risk Level | Customers | Churn Probability | Recommended Action |
|-----------|-----------|-------------------|-------------------|
| **High** | 24,513 (25%) | >70% | Immediate win-back campaign |
| **Medium** | 31,201 (32%) | 40-70% | Re-engagement campaign |
| **Low** | 40,764 (43%) | <40% | Regular retention |

---

## 💰 Business Impact

### Win-Back Campaign Simulation

**Assumptions:**
- Target: 24,513 high-risk customers
- Campaign cost: $10 per customer
- Success rate: 30%
- Customer LTV: $100

**Results:**
- Campaign cost: **$245,130**
- Saved customers: **7,354** (30% of high-risk)
- Revenue saved: **$735,400**
- Net profit: **$490,270**
- **ROI: 200%** 🎉

---

## 🛠️ Technical Stack

### Data Processing & ML
- **pandas** 2.1.4 - Data manipulation
- **numpy** 1.26.2 - Numerical computing
- **scikit-learn** 1.3.2 - ML algorithms, preprocessing
- **xgboost** 2.0.3 - Gradient boosting
- **lightgbm** 4.1.0 - Alternative boosting
- **imbalanced-learn** 0.11.0 - SMOTE for class imbalance

### Visualization
- **matplotlib** 3.8.2 - Plotting
- **seaborn** 0.13.0 - Statistical visualizations
- **plotly** 5.18.0 - Interactive plots

### Model Management
- **mlflow** 2.9.2 - Experiment tracking
- **joblib** - Model serialization

---

## 📊 Model Performance

### Classification Report (Test Set)

```
              precision    recall  f1-score   support

    Active       0.82      0.80      0.81      7753
   Churned       0.79      0.82      0.80      11543

  accuracy                           0.81     19296
```

### Confusion Matrix

```
                Predicted
             Active  Churned
Actual Active   6202    1551   (80% correct)
       Churned   2077    9466   (82% recall)
```

**Interpretation:**
- **False Positives** (1,551): Predicted churn but actually active
  - Cost: Unnecessary campaign spending
  
- **False Negatives** (2,077): Predicted active but actually churned
  - Cost: Lost customer revenue
  
**Tradeoff**: Model optimized for high recall (catch most churners) while maintaining good precision

---

## 🔄 Model Usage Examples

### Example 1: Batch Prediction

```python
from src.predict_churn import ChurnPredictor
import pandas as pd

# Load new customer data
customers = pd.read_csv('new_customers.csv')

# Initialize predictor
predictor = ChurnPredictor()

# Get predictions
predictions = predictor.predict(customers)

# View results
print(predictions[['churn_probability', 'risk_segment', 'recommended_action']].head())
```

Output:
```
   churn_probability  risk_segment          recommended_action
0              0.82    High Risk     Immediate win-back campaign
1              0.35    Low Risk      Regular retention activities
2              0.61    Medium Risk   Re-engagement campaign
```

### Example 2: Single Customer

```python
customer = {
    'recency': 150,
    'monetary': 500,
    'segment_Champions': 0,
    'segment_At_Risk': 1,
    'avg_delivery_time': 15,
    # ... other features
}

result = predictor.predict_single(customer)
print(f"Churn probability: {result['churn_probability']:.2%}")
print(f"Risk level: {result['risk_segment']}")
```

### Example 3: Top N Risky Customers

```python
# Get 100 most at-risk customers
top_100_risk = predictor.get_top_risk_customers(customers, top_n=100)

# Export for campaign team
top_100_risk.to_csv('high_priority_customers.csv', index=False)
```

---

## 📝 Actionable Recommendations

### Immediate Actions (High Risk)

1. **Personalized Win-Back Offers**
   - 15-20% exclusive discount
   - Free shipping on next order
   - "We miss you" email campaign

2. **Customer Service Outreach**
   - Personal phone call
   - Address past issues
   - Exclusive support channel

### Preventive Actions (Medium Risk)

1. **Re-Engagement Campaigns**
   - Product recommendations based on history
   - New arrivals in favorite categories
   - Limited-time offers

2. **Loyalty Program**
   - Points for purchases
   - VIP tier benefits
   - Early access to sales

### Retention (Low Risk)

1. **Regular Communication**
   - Monthly newsletters
   - Product launches
   - Content marketing

2. **Referral Program**
   - Incentivize word-of-mouth
   - Rewards for referrals

### Operational Improvements

1. **Delivery Optimization**
   - Target: <10 days average
   - Real-time tracking
   - Delivery guarantees

2. **Customer Satisfaction**
   - Post-purchase surveys
   - Quick issue resolution
   - Quality control

3. **Proactive Monitoring**
   - Alert when customer reaches 90 days inactive
   - Segment-specific strategies
   - A/B test campaign effectiveness

---

## 🔮 Future Enhancements

### Model Improvements
- [ ] Try deep learning (Neural Networks)
- [ ] SHAP values for better interpretability
- [ ] Time-series features (trend analysis)
- [ ] Customer lifetime value prediction

### Feature Engineering
- [ ] Customer communication history
- [ ] Website behavior (clicks, time on site)
- [ ] Marketing channel attribution
- [ ] Seasonal patterns

### Deployment
- [ ] REST API with FastAPI
- [ ] Real-time scoring pipeline
- [ ] Model monitoring dashboard
- [ ] A/B testing framework
- [ ] Automated retraining pipeline

### Business Integration
- [ ] CRM integration
- [ ] Marketing automation connection
- [ ] Campaign ROI tracking
- [ ] Customer segmentation dashboard

---

## 📚 References

- **Dataset**: [Olist Brazilian E-Commerce](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
- **XGBoost**: [Documentation](https://xgboost.readthedocs.io/)
- **Scikit-learn**: [User Guide](https://scikit-learn.org/stable/user_guide.html)

---

## 👤 Author

**Suleyman Celik**
- GitHub: [@yourusername](https://github.com/suleyman-celik)
- LinkedIn: [Suleyman Celik](https://linkedin.com/in/yourprofile)
- Email: suleyman.clk709@gmail.com

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Olist for providing the dataset
- Kaggle for hosting the data
- Open-source ML community

---

**⭐ If you find this project useful, please star it on GitHub!**
