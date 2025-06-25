# Credit Card Fraud Detection with XGBoost

A machine learning project for detecting fraudulent credit card transactions using the XGBoost algorithm. This repository demonstrates data preprocessing, model training, evaluation, and deployment for a binary classification problem using real-world credit card transaction data.

---

## 📌 Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Tech Stack & Dependencies](#tech-stack--dependencies)
- [Installation & Setup](#installation--setup)
- [Dataset](#dataset)
- [Environment Variables](#environment-variables)
- [Available Scripts](#available-scripts)
- [Folder Structure](#folder-structure)
- [Results & Evaluation](#results--evaluation)
- [Deployment](#deployment)
- [Author](#author)

---

## 📝 Project Overview

This project uses the XGBoost algorithm to identify fraudulent credit card transactions. It focuses on imbalanced classification, effective preprocessing (including scaling and resampling), and robust evaluation techniques. The project provides a reproducible workflow from data exploration to model deployment.

---

## 🚀 Features

- **End-to-End ML Pipeline**
  - Data exploration, preprocessing, model training, evaluation, and prediction.
- **Imbalanced Data Handling**
  - Techniques such as SMOTE or random under/oversampling.
- **Explainable Results**
  - Feature importance visualization and interpretation.
- **Modular Code**
  - Easily adapt for other tabular binary classification tasks.
- **Notebooks & Scripts**
  - Jupyter notebooks for exploration and `.py` scripts for reproducibility.

---

## 🛠 Tech Stack & Dependencies

- **Language:** Python 3.8+
- **Core Libraries:** XGBoost, scikit-learn, pandas, numpy
- **Visualization:** matplotlib, seaborn
- **Jupyter:** For interactive exploration

See `requirements.txt` for the full list.

---

## 📥 Installation & Setup

1. **Clone the repository**
    ```bash
    git clone https://github.com/arifurrahmansajid/credit-card-fraud-detection-xgboost.git
    cd credit-card-fraud-detection-xgboost
    ```

2. **(Recommended) Create and activate a virtual environment**
    ```bash
    python -m venv venv
    source venv/bin/activate   # On Windows: venv\Scripts\activate
    ```

3. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4. **Download and place the dataset**
    - Download the dataset from [Kaggle: Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud).
    - Place `creditcard.csv` in the `data/` directory.

---

## 🗂 Dataset

- The dataset contains anonymized credit card transactions labeled as fraudulent or genuine.
- [Kaggle Dataset Link](https://www.kaggle.com/mlg-ulb/creditcardfraud)

---

## 🔑 Environment Variables

- If using secret keys or local paths, create a `.env` file in the project root.
- Example (not required for basic usage):
    ```
    DATA_PATH=./data/creditcard.csv
    ```

> **Note:** Ensure `.env` is included in `.gitignore`.

---

## 📋 Available Scripts

```bash
# Run exploratory data analysis
python notebooks/eda.ipynb

# Train and evaluate the model
python src/train.py

# Predict with a trained model
python src/predict.py --input example.csv
```

Or open and run the Jupyter notebooks interactively.

---

## 📂 Folder Structure

```
credit-card-fraud-detection-xgboost/
├── data/                 # Dataset (creditcard.csv)
├── notebooks/            # Jupyter Notebooks for EDA and experiments
├── src/
│   ├── __init__.py
│   ├── train.py          # Main training script
│   ├── predict.py        # Prediction script
│   ├── utils.py          # Helper functions
│   └── ...               # Additional modules
├── requirements.txt      # Python dependencies
├── .env                  # Environment variables (if any, ignored)
├── README.md             # Project documentation
└── ...                   # Other config files
```

---

## 📊 Results & Evaluation

- Includes ROC AUC, Precision-Recall, F1 Score, and Confusion Matrix.
- Feature importance plots for interpretability.
- Results are logged in the notebook and as output files.

---

## 🚀 Deployment

- The model can be exported as a `.pkl` file for use in production or as part of a REST API (Flask/FastAPI example can be added).
- For simple web demos, see the `deployment/` folder (if exists) for sample scripts.

---

## 👤 Author

**Arifur Rahman Sajid**  
GitHub: [arifurrahmansajid](https://github.com/arifurrahmansajid)

---

Happy modeling! 🚀
