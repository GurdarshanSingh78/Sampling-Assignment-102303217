# Sampling Assignment: Credit Card Fraud Detection

## 1. Project Overview
This project investigates the impact of different sampling techniques on the performance of machine learning models. Using a real-world **Credit Card Fraud Detection** dataset, we analyze whether we can maintain high model accuracy using smaller, representative samples compared to using the entire dataset.

The focus is on improving **minority class (fraud) detection** efficiency without sacrificing overall model stability.

## 2. Problem Statement
**Class Imbalance** is a major challenge in fraud detection. Fraud transactions are extremely rare compared to normal transactions. Standard ML models often achieve high accuracy by simply guessing "Non-Fraud" for every case, failing to detect actual crimes.

This project solves this by balancing the dataset and identifying which **Sampling Technique** creates the most effective training set for 5 different ML models.

## 3. Dataset Summary

| Property | Value |
| :--- | :--- |
| **Dataset Name** | Credit Card Fraud Detection |
| **Total Features** | 30 |
| **Target Variable** | Class (0 = Non-Fraud, 1 = Fraud) |
| **Original Size** | 772 rows (Highly Imbalanced) |
| **Balanced Size** | 1526 rows (Oversampled) |
| **Sample Size Used** | 385 (Calculated via Cochran’s Formula) |

## 4. Methodology

### Step 1: Data Balancing
We balanced the dataset using **Random Oversampling** to create a 50/50 split between Fraud and Non-Fraud cases.

### Step 2: Sampling Techniques
We calculated a statistically significant sample size of **385** using **Cochran's Formula** ($Z=1.96, p=0.5, e=0.05$) and applied five distinct techniques:
1.  **Simple Random Sampling:** Purely random selection.
2.  **Systematic Sampling:** Selection based on a fixed interval ($k^{th}$ index).
3.  **Stratified Sampling:** Selection that strictly maintains the class distribution.
4.  **Cluster Sampling:** Selection of random natural groupings (clusters).
5.  **Bootstrap Sampling:** Random selection with replacement.

### Step 3: Models Evaluated
We tested these samples on five different machine learning models:
* **M1:** Logistic Regression
* **M2:** Decision Tree
* **M3:** Random Forest
* **M4:** Support Vector Machine (SVM)
* **M5:** K-Nearest Neighbors (KNN)

## 5. Key Results
The table below compares the accuracy of models trained on the **Full Dataset** vs. the **Samples**.

| Model | No Sampling | Simple Random | Systematic | Stratified | Cluster | Bootstrap |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **M1 (Logistic Regression)** | 91.36% | 92.15% | 89.79% | 91.10% | 90.31% | **92.41%** |
| **M2 (Decision Tree)** | 99.48% | 98.69% | **99.48%** | 98.17% | 97.12% | 96.86% |
| **M3 (Random Forest)** | 100.00% | 99.21% | **99.74%** | **99.74%** | 99.21% | 99.21% |
| **M4 (SVM)** | 67.54% | 67.02% | 66.75% | **70.16%** | 66.23% | 66.23% |
| **M5 (KNN)** | 98.17% | 95.55% | 96.07% | **96.60%** | 93.19% | 94.76% |

## 6. Discussion
* **Winner:** **Stratified Sampling** was the most reliable technique. It achieved the highest accuracy for SVM (70.16%) and tied for the best results on Random Forest. This confirms that preserving the class ratio is crucial for small samples.
* **Efficiency:** We achieved **99.74% accuracy** (Random Forest) using only **385 rows** of data, which is computationally much faster than using the full dataset.
* **Surprise:** For Logistic Regression, the **Bootstrap Sample** actually outperformed the full dataset, suggesting it helped reduce overfitting.

## 7. Performance Graph
![Accuracy Comparison Graph](sampling_comparison_graph.png)

## 8. Technologies Used
* **Language:** Python 3.10+
* **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
* **Environment:** Google Colab / VS Code

---
Submitted By: **Gurdarshan Singh**
Roll Number: **102303217**
