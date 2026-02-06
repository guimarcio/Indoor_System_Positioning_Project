# Files to be updated!

# 📍 Indoor System Positioning: Data Science Project

## 📌 Project Summary

This project applies machine learning techniques to Wi-Fi RSSI data in order to classify indoor device location across four rooms.

Using dimensionality reduction (PCA), outlier detection (Isolation Forest), and multiple classifiers (KNN, Naive Bayes, Decision Tree, SVM), the models achieved up to **99% validation accuracy**.

The best-performing model was **RBF Kernel SVM**, trained on the full dataset without outliers.

## 💡 Introduction

### 🌎 Context

Global Positioning Systems (GPS) provide accurate positioning outdoors but perform poorly in indoor environments due to signal attenuation caused by walls and other physical structures. To address this limitation, Indoor Positioning Systems (IPS) use alternative technologies such as Wi-Fi (WPS), Bluetooth, infrared, and ultrasound to estimate location within enclosed spaces.

With the widespread adoption of smartphones, IPS applications have expanded to areas such as smart building navigation, augmented reality, security monitoring, and resource management.

This project focuses on indoor localization using Wi-Fi signals and machine learning techniques. Specifically, it aims to classify the location of a mobile device among four different rooms within an office environment in Pittsburgh, USA. 

### 📂 Dataset Description

The [dataset](https://archive.ics.uci.edu/dataset/422/wireless+indoor+localization) contains RSSI measurements provided in .txt format. It includes 2,000 measurements collected using a mobile device. Each observation records the RSSI values from seven different Wi-Fi routers (7 features). The eighth column represents the room label where the measurement was taken. The dataset is balanced, with 500 samples collected in each of the four rooms.

## ⚙️Methods

### 🧹 Data processing

Class labels were shifted to range from 0 to 3 to simplify vector-based operations. No missing values or duplicate instances were found in the dataset.

Outliers were detected using the Isolation Forest algorithm (Scikit-learn), applied separately to each class. A total of 162 anomalous samples were removed, reducing the dataset to 1,838 instances.

Afterward, all features were standardized using Z-score normalization. The original dataset (with outliers) was also standardized to allow performance comparison.

### 📊 Exploratory Data Analysis

An initial exploratory analysis was conducted using histograms for each of the seven features across the four classes. A D’Agostino-Pearson normality test indicated that approximately half of the feature distributions were close to Gaussian, while others showed deviations likely caused by multipath signal propagation effects in indoor environments.

Correlation analysis revealed a strong association between features 1 and 4 (correlation = 0.92), as well as relevant correlations between feature 1 and features 6 and 7. This suggests spatial proximity or directional similarity between certain Wi-Fi routers.

![cor_map.png](cor_map.png)

Given these correlations, Principal Component Analysis (PCA) was applied for dimensionality reduction. Results showed that projecting the data onto the first two principal components (using the covariance matrix) preserved approximately 85% of the total variance, outperforming direct feature-based dimensionality reduction.

![cov_pca.png](/Images/cov_PCA.png)

Scatter plot analysis of the principal components confirmed the presence of atypical samples in Room 1, which are likely related to multipath signal effects rather than measurement errors.

![data.png](/Images/data.png)

### 🤖 Classification Models

The following supervised learning algorithms were evaluated:

* K-Nearest Neighbors (KNN)

* Gaussian Naive Bayes

* Decision Tree

* Linear SVM

* RBF Kernel SVM

All models were implemented in Python (Scikit-learn) using Jupyter Notebook.

Hyperparameters were tuned using Grid Search (when applicable). 

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

## 💭 Results and Discussion

### 💠 K-Nearest Neighbors (KNN)

KNN performance was evaluated across different values of k (1, 3, 5, 7, 9).
The best value of k was selected based on the highest mean validation accuracy. Results are reported for cross-validation and holdout evaluation.

| Dataset Configuration              | Features | Evaluation       | Best k | Train Accuracy | Test Accuracy |
| ---------------------------------- | -------- | ---------------- | ------ | -------------- | ------------- |
| Standardized data with outliers    | 7        | Cross-Validation | 3      | 99.23%         | 98.55%        |
| Standardized data without outliers | 7        | Cross-Validation | 3      | 99.54%         | 98.96%        |
| PCA-reduced data (no outliers)     | 2        | Cross-Validation | 9      | 98.60%         | 98.09%        |
| PCA-reduced data (no outliers)     | 2        | Holdout          | 9      | 98.78%         | 95.91%        |

![knn.png](/Images/knn.png)

#### Confusion Matrix
| Actual \ Predicted | Room 0 | Room 1 | Room 2 | Room 3 |
| ------------------ | ------ | ------ | ------ | ------ |
| **Room 0**         | 98     | 0      | 2      | 0      |
| **Room 1**         | 0      | 66     | 3      | 0      |
| **Room 2**         | 0      | 7      | 89     | 0      |
| **Room 3**         | 0      | 0      | 3      | 99     |

#### Classification Metrics

| Class  | Precision | Recall | F1-Score |
| ------ | --------- | ------ | -------- |
| Room 0 | 1.000     | 0.980  | 0.990    |
| Room 1 | 0.904     | 0.957  | 0.930    |
| Room 2 | 0.918     | 0.927  | 0.922    |
| Room 3 | 1.000     | 0.971  | 0.985    |


### 📈 Gaussian Naive Bayes

The same evaluation procedure was applied to Naive Bayes.
Since this model does not require hyperparameter tuning in this context, no Grid Search was performed. Performance metrics are reported for both cross-validation and holdout.

| Dataset Configuration              | Features | Evaluation       | Train Accuracy | Test Accuracy |
| ---------------------------------- | -------- | ---------------- | -------------- | ------------- |
| Standardized data with outliers    | 7        | Cross-Validation | 98.40%         | 98.25%        |
| Standardized data without outliers | 7        | Cross-Validation | 98.76%         | 98.80%        |
| PCA-reduced data (no outliers)     | 2        | Cross-Validation | 98.55%         | 97.87%        |
| PCA-reduced data (no outliers)     | 2        | Holdout          | 98.64%         | 93.73%        |

![nbayes.png](/Images/nbayes.png)

#### Confusion Matrix

| Actual \ Predicted | Room 0 | Room 1 | Room 2 | Room 3 |
| ------------------ | ------ | ------ | ------ | ------ |
| **Room 0**         | 97     | 0      | 3      | 0      |
| **Room 1**         | 0      | 66     | 3      | 0      |
| **Room 2**         | 0      | 16     | 80     | 0      |
| **Room 3**         | 0      | 0      | 1      | 101    |

#### Classification Metrics

| Class  | Precision | Recall | F1-Score |
| ------ | --------- | ------ | -------- |
| Room 0 | 1.000     | 0.970  | 0.985    |
| Room 1 | 0.805     | 0.957  | 0.874    |
| Room 2 | 0.920     | 0.833  | 0.874    |
| Room 3 | 1.000     | 0.990  | 0.995    |


### 🌲 Decision Tree

Decision Tree hyperparameters (criterion, max_depth, max_features, min_samples_leaf) were optimized using Grid Search.
Hyperparameter adjustments were performed during cross-validation to improve generalization and reduce overfitting.

| Dataset Configuration              | Features | Evaluation       | Hyperparameters *(criterion, max_depth, max_features, min_samples_leaf)* | Train Accuracy | Test Accuracy |
| ---------------------------------- | -------- | ---------------- | ------------------------------------------------------------------------ | -------------- | ------------- |
| Standardized data with outliers    | 7        | Cross-Validation | [gini, 4, 5, 5]                                                          | 96.79%         | 96.19%        |
| Standardized data without outliers | 7        | Cross-Validation | [gini, 4, 5, 5]                                                          | 97.59%         | 97.16%        |
| PCA-reduced data (no outliers)     | 2        | Cross-Validation | [gini, 3, 2, 10]                                                         | 98.26%         | 97.27%        |
| PCA-reduced data (no outliers)     | 2        | Holdout          | [gini, 3, 2, 10]                                                         | 98.30%         | 95.64%        |

![dtree.png](/Images/dtree.png)

#### Confusion Matrix

| Actual \ Predicted | Room 0 | Room 1 | Room 2 | Room 3 |
| ------------------ | ------ | ------ | ------ | ------ |
| **Room 0**         | 98     | 0      | 1      | 1      |
| **Room 1**         | 1      | 65     | 3      | 0      |
| **Room 2**         | 0      | 9      | 86     | 1      |
| **Room 3**         | 0      | 0      | 0      | 102    |

#### Classification Metrics

| Class      | Precision | Recall | F1-Score |
| ---------- | --------- | ------ | -------- |
| **Room 0** | 0.990     | 0.980  | 0.985    |
| **Room 1** | 0.878     | 0.942  | 0.909    |
| **Room 2** | 0.956     | 0.896  | 0.925    |
| **Room 3** | 0.981     | 1.000  | 0.990    |

### 🧮 Linear SVM

For the Linear SVM model, the hyperparameter C was tuned using Grid Search.
Model performance was evaluated via cross-validation and holdout testing.

| Dataset Configuration              | Features | Evaluation       | C   | Train Accuracy | Test Accuracy |
| ---------------------------------- | -------- | ---------------- | --- | -------------- | ------------- |
| Standardized data with outliers    | 7        | Cross-Validation | 1   | 98.45%         | 98.10%        |
| Standardized data without outliers | 7        | Cross-Validation | 10  | 99.02%         | 98.75%        |
| PCA-reduced data (no outliers)     | 2        | Cross-Validation | 2.5 | 98.52%         | 98.26%        |
| PCA-reduced data (no outliers)     | 2        | Holdout          | 2.5 | 98.64%         | 95.10%        |

![svm.png](/Images/svm.png)

#### Confusion Matrix

| Actual \ Predicted  | Room 0 | Room 1 | Room 2 | Room 3 |
| ---------- | ------ | ------ | ------ | ------ |
| **Room 0** | 97     | 0      | 3      | 0      |
| **Room 1** | 0      | 66     | 3      | 0      |
| **Room 2** | 0      | 11     | 85     | 0      |
| **Room 3** | 0      | 0      | 1      | 101    |


#### Classification Metrics

| Class      | Precision | Recall | F1-Score |
| ---------- | --------- | ------ | -------- |
| **Room 0** | 1.000     | 0.970  | 0.985    |
| **Room 1** | 0.857     | 0.957  | 0.904    |
| **Room 2** | 0.924     | 0.885  | 0.904    |
| **Room 3** | 1.000     | 0.990  | 0.995    |

### 🌀 RBF Kernel SVM

The RBF Kernel SVM model was trained with hyperparameters C and γ, optimized using Grid Search.
Final results are presented based on the best configuration obtained during cross-validation.

| Dataset Configuration              | Features | Evaluation       | Hyperparameters (C, γ) | Train Accuracy | Test Accuracy |
| ---------------------------------- | -------- | ---------------- | ---------------------- | -------------- | ------------- |
| Standardized data with outliers    | 7        | Cross-Validation | [1, 0.1]               | 98.40%         | 98.20%        |
| Standardized data without outliers | 7        | Cross-Validation | [10, 1]                | 99.96%         | 98.96%        |
| PCA-reduced data (no outliers)     | 2        | Cross-Validation | [10, 0.1]              | 98.53%         | 98.42%        |
| PCA-reduced data (no outliers)     | 2        | Holdout          | [10, 0.1]              | 98.50%         | 94.82%        |

![rbfsvm.png](/Images/rbfsvm.png)

#### Confusion Matrix

| Actual \ Predicted | Room 0 | Room 1 | Room 2 | Room 3 |
| ------------------ | ------ | ------ | ------ | ------ |
| **Room 0**         | 97     | 0      | 3      | 0      |
| **Room 1**         | 0      | 66     | 3      | 0      |
| **Room 2**         | 0      | 11     | 85     | 0      |
| **Room 3**         | 0      | 0      | 2      | 100    |


#### Classification Metrics

| Class  | Precision | Recall | F1-Score |
| ------ | --------- | ------ | -------- |
| Room 0 | 1.000     | 0.970  | 0.985    |
| Room 1 | 0.857     | 0.957  | 0.904    |
| Room 2 | 0.914     | 0.885  | 0.899    |
| Room 3 | 1.000     | 0.980  | 0.990    |


 ## 🔍 Key Findings

After analyzing the experimental results, several important observations were made:

Models trained on the dataset without outliers consistently achieved higher validation accuracy compared to those trained on contaminated data.

Although PCA reduced total variance by 14.25%, models trained on the reduced dataset achieved performance very close to the full 7-feature dataset.

Hyperparameter tuning via Grid Search required manual oversight, as fully automated selection often led to overfitting and poorer validation performance.

Holdout evaluation produced lower performance compared to cross-validation, highlighting the sensitivity of small datasets to data splitting strategies.

All models achieved validation accuracies above 96%, likely due to low class overlap.
The best results (~99% validation accuracy) were obtained using the outlier-free dataset with all 7 features, particularly with:

* RBF Kernel SVM

* KNN

However, RBF SVM is preferred in practice, as KNN becomes computationally expensive with larger datasets.

Additional metrics such as Precision, Recall, and F1-score were analyzed and are important depending on the intended IPS application.
