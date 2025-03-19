# 🏀 NBA Player Prediction - Machine Learning Model  

This project utilizes a **RandomForest** machine learning model to predict the missing player in an NBA lineup based on historical data. The dataset consists of past games, where one player from the home team is randomly removed, and the model is trained to predict that missing player.  

---

## 📌 Table of Contents
- [📂 Project Overview](#-project-overview)
- [🚀 Running Instructions](#-running-instructions)
- [📦 Dependencies & Installation](#-dependencies--installation)
- [📊 Model Training & Prediction](#-model-training--prediction)

---

## 📂 Project Overview
This repository contains:
- **Model Training:** Trains a **RandomForestClassifier** on NBA historical game data.
- **Player Prediction:** Predicts the missing player in a lineup from test data.
- **Performance Metrics:** Evaluates the model’s accuracy per year and across different data splits.

### 🏆 Features:
✔ Uses **RandomForestClassifier** for prediction.  
✔ Encodes categorical player & team data efficiently.  
✔ Generates **accuracy reports** for yearly evaluation.  
✔ Provides a **detailed dataset** of predictions and statistics.  

---

## 🚀 Running Instructions
Follow these steps to set up and run the project:

### 1️⃣ Clone the Repository
Open a terminal and run:
```bash
git clone https://github.com/vaghelahetvi/nba-prediction.git
cd nba-prediction

---

### 2️⃣ Install Dependencies
Ensure you have Python installed, then install all required dependencies:
```bash
pip install -r requirements.txt

---

### 3️⃣ Train the Model
Run the training script to train the model on the NBA dataset:
```bash
python nba-player-prediction.py

📌 Output: Saves the trained model and encoders.
---

### 4️⃣ Run the Prediction Model
Once the model is trained, use it to predict missing players in test data:
```bash
python missing-player-predictor.py

📌 Output: Saves the predictions to Results/NBA_test_predictions.csv

---

### 5️⃣ View Prediction Results
To preview the predicted missing players:
```bash
less Results/NBA_test_predictions.csv

(Press q to exit)

---

### 6️⃣ View Yearly Accuracy Results
To check how many predictions were correct per year:
```bash
less Results/NBA_test_yearly_accuracy.csv

(Press q to exit)

---

### **1️⃣3️⃣ Model Training & Prediction**
```md
## 📊 Model Training & Prediction
The project follows a structured **6-step machine learning pipeline**:

### 1️⃣ Load and Combine Data
- Reads and merges multiple CSV files for training.
- Cleans missing values and selects relevant features.

### 2️⃣ Data Preprocessing
- Encodes categorical features using `LabelEncoder`.
- Standardizes numerical values and optimizes data types.

### 3️⃣ Model Training
- Uses `RandomForestClassifier` with:
  - `n_estimators=100`
  - `max_depth=20`
  - `class_weight='balanced'`
- Trains on 80% of the dataset.

### 4️⃣ Model Evaluation
- Tests on 20% of unseen data.
- Computes **accuracy score** and **classification report**.

### 5️⃣ Predictions on Test Data
- Uses the trained model to predict missing players.
- Outputs predictions in `NBA_test_predictions.csv`.

### 6️⃣ Performance Analysis
- Generates a **confusion matrix**.
- Outputs **yearly accuracy reports**.



