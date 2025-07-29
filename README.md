# Loan-Approval-Prediction
This project predicts whether a loan application will be approved or rejected based on various applicant details using supervised machine learning classification techniques.


# 🏦 Loan Approval Prediction

This project focuses on predicting loan approval using supervised machine learning classification techniques. The dataset contains information about loan applicants, including their gender, marital status, education, income, credit history, and more. The objective is to build a model that can predict whether a loan will be approved (`Y`) or not (`N`).

---

## 📂 Project Workflow

1. **Data Loading & Exploration**

   * Loaded dataset using `pandas`
   * Checked for missing values and overall structure

2. **Data Cleaning & Preprocessing**

   * Handled missing values
   * Converted categorical variables into numerical values using label encoding or mapping

3. **Feature Selection & Scaling**

   * Separated features (`X`) and target (`y`)
   * Scaled input features using `StandardScaler`

4. **Train-Test Split**

   * Used `train_test_split` to divide the dataset into training and testing sets

5. **Model Training**
   Trained the dataset using the following classification algorithms:

   * ✅ Support Vector Classifier (SVC)
   * ✅ K-Nearest Neighbors (KNN)
   * ✅ Naive Bayes
   * ( Tryed all the 3 of ML Algorithm and Taken the most accurate Algorithm)
   * naive bayes scored around 80% accurate 

6. **Model Evaluation**

   * Evaluated model performance using:

     * Accuracy Score
     * Confusion Matrix
     * Classification Report (Precision, Recall, F1-Score)

---

## 🛠️ Technologies Used

* Python
* Pandas, NumPy
* Scikit-learn (SVC, KNN, GaussianNB, metrics)
* StandardScaler (for normalization)

---

## 📈 Future Improvements

* Hyperparameter tuning using GridSearchCV
* Trying advanced models like Random Forest or XGBoost
* Model deployment using Streamlit or Flask

---

## 📁 Dataset

The dataset used is typically available on public platforms like Kaggle under "Loan Prediction Problem."
(If you're using a local CSV file, you can optionally upload it here or provide the source.)

---

## ✅ Results

All three models were successfully implemented and evaluated. Model accuracy varied based on the algorithm, with potential for improvement through tuning or more complex models.

---

## 💡 Author

Athil Chand
Feel free to fork, use, or build upon this project!
