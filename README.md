# 💼 Employee Salary Prediction (Classification)

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-orange?logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-blue?logo=pandas)
![Streamlit](https://img.shields.io/badge/Streamlit-red?logo=streamlit)

This is an end-to-end machine learning project that predicts whether an adult's income is **over $50,000** or **less than or equal to $50,000** per year. This is a binary classification task based on the popular "Adult Income" dataset from the UCI repository.

The project includes two main components:
1.  A Jupyter Notebook (`Employee_Salary_Prediction.ipynb`) with the full data cleaning, analysis, and training of a high-accuracy model (86.7%).
2.  A live web app (`app.py`) powered by a simpler, more lightweight model for easy deployment.

---

## 🚀 Live App Demo

This project is deployed as a live web application using Streamlit.

**[Click here to view the live app](YOUR_STREAMLIT_APP_LINK_HERE)**

<p align="center">
  <img src="YOUR_SCREENSHOT_LINK_HERE" alt="Streamlit App Screenshot" width="700">
</p>

*(**Note:** To get these links, you can deploy your app for free on [Streamlit Community Cloud](https://streamlit.io/cloud). Just upload your `app.py`, `simple_model.pkl`, and `simple_scaler.pkl` files to a new GitHub repo and link it to Streamlit.)*

---

## 📊 Project Workflow

The main Jupyter Notebook (`Employee_Salary_Prediction.ipynb`) follows a complete data science workflow:

1.  **Data Loading:** Loaded the `adult.csv` dataset and applied correct column names.
2.  **Data Cleaning:** Handled missing values (identified as `?` and dropped) and reset the index.
3.  **Feature Engineering:**
    * **Target Encoding:** Converted the `income` column (<=50K, >50K) to numeric (`0`, `1`).
    * **One-Hot Encoding:** Converted all categorical (text) features into numeric columns using `pd.get_dummies()`, resulting in 96 total features.
4.  **Exploratory Data Analysis (EDA):** Created visualizations to understand the relationships between features like `age`, `education`, `hours-per-week`, and the `income` class.
5.  **Preprocessing:** Split the data into 80% training and 20% testing sets and scaled all features using `StandardScaler`.
6.  **Model Training & Evaluation:** Trained and compared three classification models to find the best performer.

---

## 🏆 Model Performance

The Gradient Boosting Classifier was the most accurate model for this dataset.

| Model | Test Set Accuracy |
| :--- | :--- |
| **Gradient Boosting Classifier** | **86.7%** |
| Random Forest Classifier | 85.5% |
| Logistic Regression | 85.3% |

The best model (`GradientBoostingClassifier`) and its scaler were saved to `salary_model.pkl` and `scaler.pkl`.

---

## 🖥️ Streamlit Web App

To create a fast and simple web app, a separate, lightweight model was built (`simple_model.pkl`). This model uses only 5 key features (`age`, `educational-num`, `capital-gain`, `capital-loss`, `hours-per-week`) and achieved an **81.6% accuracy**.

The `app.py` file uses this simple model to power the interactive web app.

### How to Run Locally

1.  Make sure you have `app.py`, `simple_model.pkl`, and `simple_scaler.pkl` in the same folder.
2.  Open a terminal in that folder.
3.  Install the required libraries:
    ```bash
    pip install streamlit pandas scikit-learn joblib
    ```
4.  Run the app:
    ```bash
    streamlit run app.py
    ```
