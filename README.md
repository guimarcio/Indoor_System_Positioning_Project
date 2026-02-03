# Files to be updated!

# 📍 Indoor System Positioning: Data Science Project

## 💡 Introduction

### 🌎 Context

Global Positioning Systems (GPS) provide accurate positioning outdoors but perform poorly in indoor environments due to signal attenuation caused by walls and other physical structures. To address this limitation, Indoor Positioning Systems (IPS) use alternative technologies such as Wi-Fi (WPS), Bluetooth, infrared, and ultrasound to estimate location within enclosed spaces.

With the widespread adoption of smartphones, IPS applications have expanded to areas such as smart building navigation, augmented reality, security monitoring, and resource management.

This project focuses on indoor localization using Wi-Fi signals and machine learning techniques. Specifically, it aims to classify the location of a mobile device among four different rooms within an office environment in Pittsburgh, USA. 

### 📂 Dataset Description

The [dataset](https://archive.ics.uci.edu/dataset/422/wireless+indoor+localization) is composed of RSSI (Received Signal Strength Indicator) measurements and it was provided in .txt format. It also contains 2,000 measurements collected using a mobile device. Each observation records the RSSI values from seven different Wi-Fi routers (7 features). The eighth column represents the room label where the measurement was taken. The dataset is balanced, with 500 samples collected in each of the four rooms.

## ⚙️Methods

### 🧹 Data processing

Class labels were shifted to range from 0 to 3 to simplify vector-based operations. No missing values or duplicate instances were found in the dataset.

Outliers were detected using the Isolation Forest algorithm (Scikit-learn), applied separately to each class. A total of 162 anomalous samples were removed, reducing the dataset to 1,838 instances.

Afterward, all features were standardized using Z-score normalization. The original dataset (with outliers) was also standardized to allow performance comparison.

### 📊 Exploratory Data Analysis

An initial exploratory analysis was conducted using histograms for each of the seven features across the four classes. A D’Agostino-Pearson normality test indicated that approximately half of the feature distributions were close to Gaussian, while others showed deviations likely caused by multipath signal propagation effects in indoor environments.

Correlation analysis revealed a strong association between features 1 and 4 (correlation = 0.92), as well as relevant correlations between feature 1 and features 6 and 7. This suggests spatial proximity or directional similarity between certain Wi-Fi routers.

Given these correlations, Principal Component Analysis (PCA) was applied for dimensionality reduction. Results showed that projecting the data onto the first two principal components (using the covariance matrix) preserved approximately 85% of the total variance, outperforming direct feature-based dimensionality reduction.

Scatter plot analysis of the principal components confirmed the presence of atypical samples in Room 1, which are likely related to multipath signal effects rather than measurement errors.

### 🤖 Classification Models

The following supervised learning algorithms were evaluated:

* K-Nearest Neighbors (KNN)

* Gaussian Naive Bayes

* Decision Tree

* Linear SVM

* RBF Kernel SVM

All models were implemented in Python (Scikit-learn) using Jupyter Notebook.

Hyperparameters were tuned using Grid Search (when applicable). For KNN, different values of k (1, 3, 5, 7, 9) were tested. Decision Tree and SVM models had their main hyperparameters optimized automatically. For the RBF SVM, both C and γ were tuned.

Model evaluation was performed using 5-fold cross-validation, with an 80/20 train-validation split. Mean training and validation accuracy were used as primary metrics.

Additionally, a holdout evaluation (80% training, 20% testing) was conducted using the PCA-transformed dataset (2 principal components, preserving ~85.75% of total variance). This allowed computation of:

* Confusion matrix

* Precision, Recall, and F1-score

* Decision boundary visualization

Three dataset configurations were compared during cross-validation:

* Standardized data with outliers

* Standardized data without outliers

* PCA-reduced data (without outliers)

The final holdout evaluation was performed using the PCA-reduced dataset.

