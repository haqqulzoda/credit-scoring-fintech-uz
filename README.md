# 🏦 Transparent Credit Scoring API (Uzbekistan Fintech Context)

## 📌 Project Overview
This project implements a **Transparent Credit Scoring System** that not only predicts loan default probability but explains *why* a decision was made using SHAP (Shapley Additive exPlanations).

**Live Demo:** [https://credit-scoring-fintech-uz.streamlit.app/]

## 🔧 Tech Stack
* **Core:** Python 3.9, Pandas, NumPy
* **ML:** XGBoost (Gradient Boosting), Scikit-Learn
* **Explainability:** SHAP (TreeExplainer)
* **App:** Streamlit, Docker ready
* **Ops:** Joblib serialization

## 📊 Key Results
| Model | ROC-AUC | Recall (Bad Risk) | Verdict |
|-------|---------|-------------------|---------|
| Logistic Regression | 0.76 | 45% | Baseline |
| Random Forest | 0.66 | 38% | Stable but weak recall |
| **XGBoost (Selected)** | **0.69** | **52%** | **Best balance of Risk Detection** |

*Note: The Champion Model prioritizes Recall (catching bad loans) while maintaining interpretability.*

## 🚀 How to Run Locally
1. Clone the repo:
   ```bash
   git clone https://github.com/haqqulzoda/credit-scoring-fintech-uz.git

2. Install dependencies:
   ```bash
   pip install -r requirements.txt

3. Run the App
   ```bash
   streamlit run src/app.py

📂 Project Structure
  * `src/app.py`: The frontend application.

  * `notebooks/`:

      * `01_eda.ipynb`: Discovery and cleaning.

      * `02_preprocessing.ipynb`: Pipeline construction.

      * `03_modeling.ipynb`: Training and SHAP analysis.
  
  * `models/`: Serialized XGBoost model and Preprocessor.

⚠️ Limitations (MVP)
  * Dataset: Uses "German Credit Data" as a proxy for financial behavior.

  * Input: The web demo simplifies inputs (only 5 key features) for UX; the backend handles 20 features using median imputation.
