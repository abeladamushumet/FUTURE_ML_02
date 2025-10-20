# Customer Churn Prediction System  

## Overview  
This project builds a **Machine Learning system to predict customer churn** — identifying which customers are likely to stop using a service.  
It combines **data analysis, predictive modeling, and visualization** to help businesses reduce churn and improve retention strategies.  

---

## Objectives  
- Predict customer churn probability.  
- Identify key drivers influencing churn.
- Present insights using **Power BI dashboard** and **Streamlit app**.  
- Deliver an end-to-end data science pipeline from raw data to deployment.  

---

## Tech Stack  
| Category | Tools / Libraries |
|-----------|-------------------|
| Language | Python |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Power BI |
| Modeling | Scikit-learn, XGBoost |
| Deployment | Streamlit |
| Environment | Virtualenv / VS Code |

---

## Project Structure  

FUTURE_ML_02/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/Telco-Customer-Churn.csv
│   └── processed/churn_cleaned.csv
│
├── notebooks/
│   ├── data_exploration.ipynb
│   ├── feature_engineering.ipynb
│   └── model_training.ipynb
│
├── model/churn_model.pkl
├── app/app.py
└── report/churn_dashboard.pbix


---

## Setup Instructions   

### 1. Clone the Repository  
```bash
git clone https://github.com/<abeladamushumet>/FUTURE_ML_02.git
cd FUTURE_ML_02
```

### 2. Create and Activate Virtual Environment  
```bash
python -m venv myenv
myenv\Scripts\activate     
```

### 3. Install Dependencies  
```bash
pip install -r requirements.txt
```

### 4. Run Notebooks  
Explore the steps in the `notebooks/` folder:  

- `data_exploration.ipynb` → Understand dataset  
- `feature_engineering.ipynb` → Clean & encode data  
- `model_training.ipynb` → Train models and Evaluate 

### 5. Launch Streamlit App
```bash
streamlit run app/app.py
```

---

##  Modeling Steps  

1. **Data Exploration:** Check distribution, missing values, and churn patterns.  
2. **Feature Engineering:** Encode categorical data and scale numerical features.  
3. **Model Training:** Compare Logistic Regression, Random Forest, and XGBoost.  
4. **Evaluation:** Use accuracy, recall, precision, F1-score, and ROC-AUC.  
5. **Visualization:** Identify churn drivers and create business insights.  

---

## Dashboard  

- **Tool:** Power BI  
- **File:** `report/churn_dashboard.pbix`  
- Displays churn by customer segments, payment methods, and tenure.  

---

---

## Results & Insights  

- Key churn drivers identified from model feature importance.  
- Business dashboard summarizes actionable insights.  
- Model can predict churn probability for each customer.  

---

## License  
This project is licensed under the **Apach License** — see the [LICENSE](LICENSE) file for details.  

---