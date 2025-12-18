# ❤️ HeartDisease-Model

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![GitHub issues](https://img.shields.io/github/issues/Sahil-Shrivas/HeartDisease-Model)](https://github.com/Sahil-Shrivas/HeartDisease-Model/issues)

A machine learning project to **predict heart disease** using patient clinical and diagnostic data. The model leverages Python and popular ML libraries to estimate risk and assist in early diagnosis.

---

## 📖 Overview

Heart disease is one of the leading causes of death worldwide. Early detection is critical to improving patient outcomes. This project builds a **predictive model** to identify the likelihood of heart disease using clinical features such as:

- Age, sex, chest pain type
- Resting blood pressure
- Cholesterol levels
- Fasting blood sugar
- Electrocardiographic results
- Maximum heart rate achieved
- Exercise-induced angina
- ST depression
- Slope of ST segment
- Number of major vessels colored by fluoroscopy
- Thalassemia

The model classifies patients into **Heart Disease: Yes / No** using supervised machine learning techniques.

---

## 🛠️ Tech Stack & Libraries

- **Language:** Python  
- **Libraries & Tools:**  
  - `pandas`, `numpy` — data manipulation  
  - `scikit-learn` — model training and evaluation  
  - `matplotlib`, `seaborn` — data visualization  
  - `pickle` / `joblib` — model persistence  
  - Optional: `streamlit` for interactive UI  

> Check `requirements.txt` for the full list of dependencies.

---

## 📂 Project Structure

    HeartDisease-Model/
    │── data/ # Dataset (raw or processed)
    │── notebooks/ # Exploratory Data Analysis & model experimentation
    │── model/ # Saved trained model file(s)
    │── app.py / predict.py # Scripts for inference / UI
    │── requirements.txt # Dependencies
    │── README.md # Project documentation
    │── LICENSE # MIT License


---

## 🏗️ Features & Functionality

- Load and preprocess the dataset: handle missing values, encode categorical features, scale numerical data  
- Split data into training and testing sets  
- Train multiple classification models: Logistic Regression, Decision Tree, Random Forest  
- Evaluate model performance using:  
  - Accuracy  
  - Precision & Recall  
  - F1-Score  
  - Confusion Matrix  
- Optional: Web-based prediction using Streamlit  

> ⚠️ Medical models should be used as a reference only. Predictions are **not a substitute for professional diagnosis**.

---

## 📊 Dataset

The dataset contains **303 patient records** with **14 attributes** related to heart health. Features include demographic, clinical, and diagnostic measurements.  

> Note: Ensure data privacy when handling real patient data.

---

## 📊 Model Functionality

  -Loads and preprocesses the dataset (handles missing values, encodes categorical variables, scales numeric features)

  -Splits data into training and testing sets

  -Trains classification algorithms (Logistic Regression, Decision Tree, Random Forest, etc.)

  -Evaluates model performance using metrics such as accuracy, precision, recall, F1-score, and confusion matrix

---

## 🚀 How to Run

1. **Clone the repository**  
    ```bash
    git clone https://github.com/Sahil-Shrivas/HeartDisease-Model.git
    cd HeartDisease-Model

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   
3. **Run the prediction script / app**
   ```bash
   python app.py

---

## Screenshot

![alt text](https://raw.githubusercontent.com/Sahil-Shrivas/HeartDisease_Model-Project/refs/heads/main/Screenshot%202025-12-18%20204501.png)
![alt text](https://raw.githubusercontent.com/Sahil-Shrivas/HeartDisease_Model-Project/refs/heads/main/Screenshot%202025-12-18%20204511.png)
![alt text](https://raw.githubusercontent.com/Sahil-Shrivas/HeartDisease_Model-Project/refs/heads/main/Screenshot%202025-12-18%20204526.png)
![alt text](https://raw.githubusercontent.com/Sahil-Shrivas/HeartDisease_Model-Project/refs/heads/main/Screenshot%202025-12-18%20204534.png)
![alt text](https://raw.githubusercontent.com/Sahil-Shrivas/HeartDisease_Model-Project/refs/heads/main/Screenshot%202025-12-18%20204543.png)
![alt text](https://raw.githubusercontent.com/Sahil-Shrivas/HeartDisease_Model-Project/refs/heads/main/Screenshot%202025-12-18%20204553.png)
![alt text](https://raw.githubusercontent.com/Sahil-Shrivas/HeartDisease_Model-Project/refs/heads/main/Screenshot%202025-12-18%20204602.png)
![alt text](https://raw.githubusercontent.com/Sahil-Shrivas/HeartDisease_Model-Project/refs/heads/main/Screenshot%202025-12-18%20205024.png)
![alt text](https://raw.githubusercontent.com/Sahil-Shrivas/HeartDisease_Model-Project/refs/heads/main/Screenshot%202025-12-18%20205059.png)

---

## 📬 Contact

- Author: Sahil Shrivas
- GitHub: https://github.com/Sahil-Shrivas

---


