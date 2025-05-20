# Insurance Policy Holder Prediction using Random Forest

## 🧠 Project Overview

This project aims to predict whether a customer will purchase an insurance policy based on various demographic and behavioral features. The model uses the **Random Forest Classifier** and optimizes hyperparameters using **GridSearchCV** to improve accuracy and performance.

## ✅ Problem Statement

Insurance companies often seek to identify potential policyholders who are likely to purchase their insurance plans. By predicting customer behavior using historical data, companies can improve targeting strategies and marketing efficiency.

## 🔍 Objective

- Build a machine learning model to classify customers as potential insurance policy purchasers.
- Use **Random Forest** algorithm for classification.
- Optimize the model with **GridSearchCV** for the best performance.

---

## 📁 Project Structure
insurance-policy-prediction/
│
├── data/
│ └── insurance_dataset.xlsx
│
├── notebook/
│ └── Insurance_Prediction.ipynb
│
├── model/
│ ├── PCA Analysis.py
│ └── Random_forest_algorithm.py
│ └── Tuned_model.py

│
├── README.md


## ⚙️ Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn (for EDA and visualization)
- Jupyter Notebook
  
Example features (customize based on your dataset):

- Age
- Gender
- Vehicle type
- Region
- Previous insurance history
- Annual income
- Claim history

Target variable: `Purchased` (Yes/No or 1/0)

---

## 🚀 Model Building Steps

### Data Preprocessing
- Handled missing values
- Encoded categorical variables
- Feature scaling (if needed)
- Train-test split (e.g., 80/20)

📈 Results
Base Random Forest Accuracy: 94.53%

Tuned Random Forest Accuracy (GridSearchCV): 95.02%

Improved precision/recall on minority class

ROC-AUC Score: 97%

📌 Future Enhancements
- Include more features (behavioral or transactional)
- Add XGBoost or LightGBM for comparison
- Deploy model with Flask or Streamlit

💡 Conclusion
The Random Forest algorithm, when properly tuned using GridSearchCV, can significantly enhance prediction performance for identifying potential insurance policy holders. This enables businesses to make data-driven marketing decisions and improve ROI.

📬 Contact
For queries or contributions, contact [Hanirudh Ravulapalli] at [hanirudhravulapalli18@gmail.com]

Let me know if you want a version tailored for deployment with Streamlit or Flask, or a dataset-specific version!

