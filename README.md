# High Accuracy COVID-19 Prediction Using Optimized Union Ensemble Feature Selection Approach

This repository contains implementation of the above research paper, which uses a union ensemble feature selection method followed by Hyperparameter optimization using a genetic algorithm to predict COVID-19.

## Data Preprocessing Steps

### 1. Dataset Loading and Initial Inspection

- Loaded the original `master_dataset.csv` file
- Fixed column alignment issues in the original dataset
- The original dataset has 59 columns.
- Select 27 relevant columns for analysis

### 2. Feature Selection

Selected 27 key features for analysis, including:
- Demographic factors: sex, age, BMI
- Health behaviors: smoking, alcohol, cannabis, amphetamines, cocaine
- Social factors: contacts count, working environment
- Risk reduction behaviors: mask usage, social distancing, single, covid19_contact
- Pre-existing medical conditions: asthma, heart disease, diabetes, lung diseases, compromised immune system, hiv_positive, hypertension, kidney diseases, other chronic disesases
- Target variable: COVID-19 positive status

### 3. Data Cleaning and Transformation

#### Handling Age Data
- Converted age range strings (e.g., '20_30') to numerical values using the average (25)
- Filled missing age values with the mean

#### Handling Missing Values
- Filled categorical missing values with mode (most frequent value)
- Filled numerical missing values with mean

#### Categorical Variable Encoding
- Applied one-hot encoding to nominal variables:
  - Sex (male, female, other, undefined)
  - Smoking status (never, quit, vape, light/medium/heavy)
  - Working environment (home, never, stopped, travel critical, travel non-critical)

#### Numerical Variable Processing
- Converted drug use columns to numeric format
- Applied Min-Max scaling to normalize all features

### 4. Class Imbalance Handling

- Applied SMOTE (Synthetic Minority Over-sampling Technique) to balance the dataset
- Created a 1:3 ratio of positive to negative COVID-19 cases
- Resulted in a balanced dataset with 1,348,342 samples

## Final Dataset

The final preprocessed dataset contains 41 features (after one-hot encoding) and is ready for machine learning model development.

## Features in Final Dataset

- Normalized numerical values (age, BMI, alcohol consumption, etc.)
- Binary health indicators (asthma, diabetes, etc.)
- One-hot encoded categorical variables
- Target variable: COVID-19 positive status (0 or 1)

## Data Preparation

The dataset was preprocessed to handle missing values with SMOTE applied for class imbalance and encoding for categorical variables. The processed dataset was split into:
- 70% training data
- 15% validation data
- 15% test data

Features were scaled using StandardScaler for applicable models.

## Feature Selection Methods

Three feature selection techniques were implemented, each selecting the top 15 features:

1. **MIFS (Mutual Information Feature Selection)**
   - Features: bmi, contacts_count, age, alcohol, rate_reducing_risk_single, rate_reducing_mask, covid19_symptoms, cannabis, covid19_contact, sex_male, sex_female, working_stopped, working_never, smoking_never, working_travel critical

2. **RFE (Recursive Feature Elimination)**
   - Features: age, alcohol, cannabis, contacts_count, rate_reducing_risk_single, covid19_symptoms, covid19_contact, asthma, other_chronic, nursing_home, sex_female, sex_male, smoking_quit10, smoking_yesmedium, working_stopped

3. **RidgeCV-based Feature Selection**
   - Features: covid19_symptoms, age, rate_reducing_risk_single, covid19_contact, alcohol, sex_male, sex_female, nursing_home, contacts_count, working_never, cannabis, heart_disease, working_stopped, asthma, smoking_yesmedium

Additionally, a **Union Ensemble** approach was utilized, combining features from all three methods, resulting in 21 unique features.

## Models Evaluated

Four classification models were trained and evaluated:
- Linear SVM (LSVM)
- Logistic Regression
- Gradient Boosting
- AdaBoost

## Results

### Table: Training on all features (no feature selection)

| Model | Validation Accuracy | Test Accuracy | Test Precision | Test Recall | Test F1 | Test AUC |
|-------|---------------------|---------------|----------------|-------------|---------|----------|
| LSVM | 0.8698 | 0.8690 | 0.8465 | 0.5815 | 0.6894 | N/A |
| LogisticRegression | 0.8677 | 0.8669 | 0.8201 | 0.5991 | 0.6924 | 0.9034 |
| GradientBoosting | 0.9599 | 0.9594 | 0.9591 | 0.8749 | 0.9151 | 0.9889 |
| AdaBoost | 0.9296 | 0.9288 | 0.9156 | 0.7880 | 0.8470 | 0.9697 |

### Feature Selection Method Comparison

### Table 1: Training on MIFS FS Subset

| Model | Validation Accuracy | Test Accuracy | Test Precision | Test Recall | Test F1 | Test AUC |
|-------|---------------------|---------------|----------------|-------------|---------|----------|
| LSVM | 0.8690 | 0.8678 | 0.8449 | 0.5773 | 0.6859 | N/A |
| LogisticRegression | 0.8664 | 0.8655 | 0.8180 | 0.5944 | 0.6885 | 0.9005 |
| GradientBoosting | 0.9608 | 0.9603 | 0.9585 | 0.8791 | 0.9171 | 0.9884 |
| AdaBoost | 0.9258 | 0.9254 | 0.9082 | 0.7805 | 0.8395 | 0.9700 |

### Table 2: Training on RFE FS Subset

| Model | Validation Accuracy | Test Accuracy | Test Precision | Test Recall | Test F1 | Test AUC |
|-------|---------------------|---------------|----------------|-------------|---------|----------|
| LSVM | 0.8699 | 0.8687 | 0.8475 | 0.5790 | 0.6880 | N/A |
| LogisticRegression | 0.8675 | 0.8663 | 0.8208 | 0.5950 | 0.6899 | 0.9017 |
| GradientBoosting | 0.9595 | 0.9585 | 0.9525 | 0.8778 | 0.9136 | 0.9868 |
| AdaBoost | 0.9305 | 0.9300 | 0.9147 | 0.7938 | 0.8500 | 0.9693 |

### Table 3: Training on RidgeCV FS Subset

| Model | Validation Accuracy | Test Accuracy | Test Precision | Test Recall | Test F1 | Test AUC |
|-------|---------------------|---------------|----------------|-------------|---------|----------|
| LSVM | 0.8697 | 0.8685 | 0.8467 | 0.5788 | 0.6876 | N/A |
| LogisticRegression | 0.8674 | 0.8664 | 0.8208 | 0.5957 | 0.6904 | 0.9011 |
| GradientBoosting | 0.9605 | 0.9596 | 0.9541 | 0.8807 | 0.9160 | 0.9872 |
| AdaBoost | 0.9302 | 0.9294 | 0.9129 | 0.7933 | 0.8490 | 0.9693 |

### Table 1: Training on union of MIFS and RFE features

| Model | Validation Accuracy | Test Accuracy | Test Precision | Test Recall | Test F1 | Test AUC |
|-------|---------------------|---------------|----------------|-------------|---------|----------|
| LSVM | 0.8703 | 0.8691 | 0.8485 | 0.5798 | 0.6888 | N/A |
| LogisticRegression | 0.8672 | 0.8662 | 0.8200 | 0.5954 | 0.6899 | 0.9024 |
| GradientBoosting | 0.9581 | 0.9574 | 0.9579 | 0.8679 | 0.9107 | 0.9877 |
| AdaBoost | 0.9296 | 0.9288 | 0.9156 | 0.7880 | 0.8470 | 0.9697 |

### Table 2: Training on union of MIFS and RidgeCV features

| Model | Validation Accuracy | Test Accuracy | Test Precision | Test Recall | Test F1 | Test AUC |
|-------|---------------------|---------------|----------------|-------------|---------|----------|
| LSVM | 0.8699 | 0.8686 | 0.8481 | 0.5780 | 0.6875 | N/A |
| LogisticRegression | 0.8675 | 0.8662 | 0.8201 | 0.5955 | 0.6900 | 0.9016 |
| GradientBoosting | 0.9600 | 0.9594 | 0.9592 | 0.8746 | 0.9150 | 0.9881 |
| AdaBoost | 0.9296 | 0.9288 | 0.9156 | 0.7880 | 0.8470 | 0.9697 |

### Table 3: Training on union of RFE and RidgeCV features

| Model | Validation Accuracy | Test Accuracy | Test Precision | Test Recall | Test F1 | Test AUC |
|-------|---------------------|---------------|----------------|-------------|---------|----------|
| LSVM | 0.8703 | 0.8689 | 0.8472 | 0.5804 | 0.6889 | N/A |
| LogisticRegression | 0.8678 | 0.8668 | 0.8218 | 0.5964 | 0.6912 | 0.9019 |
| GradientBoosting | 0.9595 | 0.9585 | 0.9525 | 0.8778 | 0.9136 | 0.9868 |
| AdaBoost | 0.9305 | 0.9300 | 0.9147 | 0.7938 | 0.8500 | 0.9693 |





### Union Ensemble Feature Selection

| Model | Test Accuracy | Test Precision | Test Recall | Test F1 | Test AUC |
|-------|--------------|---------------|------------|---------|----------|
| LSVM | 0.8691 | 0.8475 | 0.5810 | 0.6894 | N/A |
| LogisticRegression | 0.8665 | 0.8206 | 0.5963 | 0.6907 | 0.9025 |
| GradientBoosting | 0.9574 | 0.9579 | 0.8679 | 0.9107 | 0.9877 |
| AdaBoost | 0.9288 | 0.9156 | 0.7880 | 0.8470 | 0.9697 |

### PCA Feature Extraction (15 components)

| Model | Test Accuracy | Test Precision | Test Recall | Test F1 | Test AUC |
|-------|--------------|---------------|------------|---------|----------|
| LSVM | 0.8476 | 0.8132 | 0.5067 | 0.6244 | N/A |
| LogisticRegression | 0.8499 | 0.7929 | 0.5410 | 0.6432 | 0.8851 |
| GradientBoosting | 0.8652 | 0.8409 | 0.5683 | 0.6783 | 0.8989 |
| AdaBoost | 0.8558 | 0.7970 | 0.5678 | 0.6632 | 0.8870 |

## Key Findings

1. **Gradient Boosting** consistently performed best across all feature subsets, achieving over 95% accuracy and F1 scores above 0.91.
2. **Feature selection** maintained or slightly improved model performance while reducing dimensionality.
3. The **Union Ensemble** approach provided comparable performance to using all features.
4. **PCA** performed worse than the other feature selection methods, suggesting that the original features contain important information that may be lost in the transformation.
5. Important features for COVID-19 prediction include: covid19_symptoms, age, contacts_count, alcohol consumption, and covid19_contact.


## Genetic Algorithm Hyperparameter Optimization

The project uses a genetic algorithm framework to search for optimal hyperparameters across different machine learning models. After optimization, the best-performing models are evaluated on test data and analyzed using SHAP (SHapley Additive exPlanations) to understand feature importance.

## Components

### Genetic Algorithm Framework

The `ga_hpo_model` function provides a generic GA implementation for hyperparameter optimization with:
- Random individual generation
- Fitness evaluation
- Selection based on fitness scores
- Crossover between selected individuals
- Mutation of offspring
- Generational evolution tracking

### Optimized Models

Four classification algorithms are optimized:

1. **AdaBoost**
   - Hyperparameters: `n_estimators` (50-200), `learning_rate` (0.01-1.0)
   - Best parameters: `n_estimators=184`, `learning_rate=0.978`
   - Validation accuracy: 0.9608
   - Test accuracy: 0.9603

2. **Gradient Boosting**
   - Hyperparameters: `n_estimators` (50-200), `learning_rate` (0.01-1.0)
   - Best parameters: `n_estimators=188`, `learning_rate=0.883`
   - Validation accuracy: 0.9907
   - Test accuracy: 0.9902

3. **Linear SVM**
   - Hyperparameters: `C` (0.001-100)
   - Best parameters: `C=9.749`
   - Validation accuracy: 0.8704
   - Test accuracy: 0.8691

4. **Logistic Regression**
   - Hyperparameters: `C` (0.001-100)
   - Best parameters: `C=0.626`
   - Validation accuracy: 0.8676
   - Test accuracy: 0.8665

### SHAP Analysis

The code includes SHAP analysis for all optimized models to explain their predictions:
- KernelExplainer is used for model-agnostic explanations
- Various visualization types:
  - Beeswarm summary plots for feature impact distribution
  - Bar plots for global feature importance

## Results

Gradient Boosting achieved the highest accuracy (99.02% on test data), significantly outperforming other models:
Here's a simple table for the final test accuracy results:

| Model | Final Test Accuracy |
|-------|---------------------|
| Gradient Boosting | 0.9902 |
| AdaBoost | 0.9603 |
| Linear SVM | 0.8691 |
| Logistic Regression | 0.8665 |

## Visualization

SHAP visualizations reveal which features contribute most to each model's predictions, providing interpretability alongside performance optimization.

### Adaboost

![image](https://github.com/user-attachments/assets/0bf1ce8c-e979-4edc-9fca-e08cee6b6b6a)

![image](https://github.com/user-attachments/assets/133e9bab-c1d9-4694-a0e6-03d003083e40)

### Gradient Boosting
![image](https://github.com/user-attachments/assets/f07c4734-7e9b-4998-80fa-2eabcb15beea)
![image](https://github.com/user-attachments/assets/ccb66ecb-4a0f-477c-84f2-403f1a57aa3e)

### LSVM

![image](https://github.com/user-attachments/assets/fd7aed65-f980-453d-a898-abdcce9969d1)

![image](https://github.com/user-attachments/assets/0f6bc780-4b5c-4c93-a0e9-cd66a55c1b7a)

### Logistic Regression

![image](https://github.com/user-attachments/assets/9e8d8360-d40b-45bf-9ae4-491c2d945a5b)

![image](https://github.com/user-attachments/assets/9ae67147-185b-45fb-9fb1-f492f25224e8)

# Novelty 
## COVID-19 Prediction using Autoencoder + ANN

Predict COVID-19 status using a cleaned, balanced dataset, with dimensionality reduction through a deep **autoencoder**, and classification using a carefully designed **artificial neural network (ANN)**. The model trains **6x faster** (2 hours vs 12 hours).

##  Methodology

### 1. Outlier Removal

Outliers are values that are too far from the normal range and can confuse the model.

We used the **Interquartile Range (IQR)** method:

- For each feature:
  - Calculate Q1 (25th percentile) and Q3 (75th percentile)
  - IQR = Q3 - Q1
  - Remove values that fall outside the range:  
    `Q1 - 1.5*IQR` to `Q3 + 1.5*IQR`

This helps in:
- Removing noisy data
- Making the training faster and more stable

### 2.  Feature Scaling

We used **Min-Max Normalization** to scale all values between 0 and 1.

### 3. Data Balancing with SMOTEENN

To fix imbalance, we used **SMOTEENN**:

- **SMOTE**: Adds synthetic samples for the minority class
- **ENN (Edited Nearest Neighbors)**: Removes overlapping or noisy data points

## Deep Autoencoder for Feature Reduction

### What is an Autoencoder?

An **autoencoder** is a neural network that:
- **Compresses** input data into a smaller format (**encoder**)
- Then tries to **reconstruct** it back to the original (**decoder**)

This helps in:
- **Reducing the number of input features**
- Removing noise and focusing on meaningful patterns



###  Autoencoder Architecture

```
Input Layer(27)  → Dense(128) → LeakyReLU → BatchNorm  
             → Dense(64)  → LeakyReLU → BatchNorm  
             → Dense(15)  → LeakyReLU (This is the compressed vector)

             → Dense(64)  → LeakyReLU → BatchNorm  
             → Dense(128) → LeakyReLU → BatchNorm  
             → Output Layer (same size as input)
```

![128](https://github.com/user-attachments/assets/27f410cd-f82a-4894-8639-63763f08718b)


**Output of the encoder (15 values)** is what we used to train our ANN — it’s like compressing all original features into just 15 numbers that still carry all important information.



## ANN for Final Classification

After compressing the data with the autoencoder, we fed it into a **ANN** to predict whether someone is COVID-positive.

###  ANN Architecture

```
Input (15)  
→ Dense(128) → LeakyReLU → BatchNorm → Dropout(0.3)  
→ Dense(64)  → LeakyReLU → BatchNorm → Dropout(0.2)  
→ Dense(32)  → LeakyReLU → BatchNorm → Dropout(0.1)  
→ Output (1 node, Sigmoid activation for binary classification)
```

- **LeakyReLU**: Better than standard ReLU for learning
- **BatchNorm**: Keeps learning stable
- **Dropout**: Helps prevent overfitting
- **Sigmoid Output**: Gives probability of COVID-positive


##  Final Model Performance

| Metric       | Value       |
|--------------|-------------|
| Accuracy     | 96.77%      |
| Precision    | 94.03%      |
| Recall       | 99.96%      |
| F1 Score     | 96.90%      |
| AUC (ROC)    | ~0.99       |

**Confusion Matrix:**
```
[[19269  1335]
 [    8 21033]]
```

**ROC Curve**

![image](https://github.com/user-attachments/assets/dec94d01-9b54-4d61-8adc-d5f9577f6728)


# COVID-19 Prediction using TabNet

## Methodology

### 1. Boolean Conversion

All boolean columns in the dataset are converted into integer format to make them compatible with the model, as machine learning models require numerical inputs.

### 2. Outlier Removal

Outliers in numerical features are removed using the Interquartile Range (IQR) method. This helps reduce noise and improve the performance and generalization of the model.

### 3. Normalization

All input features (except the target column) are standardized using Z-score normalization. This ensures that each feature contributes equally during model training by scaling them to a standard range.

### 4. Handling Class Imbalance

The dataset is likely imbalanced (i.e., more negative samples than positive ones). SMOTE (Synthetic Minority Over-sampling Technique) is used to generate synthetic samples of the minority class. This balances the dataset and helps prevent the model from being biased toward the majority class.

### 5. Train-Test Split

The balanced and normalized data is split into training and testing sets using stratified sampling to ensure that the class distribution is maintained in both subsets.

### 6. Model Selection: TabNet

TabNetClassifier is used for training. TabNet is a deep learning architecture designed specifically for tabular datasets. It uses sequential attention to select relevant features for each decision step, improving interpretability and efficiency.

The model is trained with early stopping to prevent overfitting. Training stops if the validation performance doesn’t improve for a set number of epochs.

### 7. Evaluation


The model achieves strong performance:
- Accuracy: ~97.17%
- Precision: ~95.56%
- Recall: ~98.93%
- F1 Score: ~97.22%
- AUC: ~0.993

### ROC Curve

![image](https://github.com/user-attachments/assets/9213f73b-ed5f-48dd-8f9c-19ae13b73337)

### Confusion Matrix

![image](https://github.com/user-attachments/assets/bc0beb6d-da9b-4e0c-9031-dbefe70f70ca)

---

# COVID-19 Prediction using Stacking Ensemble

Predict COVID-19 positive cases by combining multiple machine learning models using a stacking ensemble approach. It includes steps for preprocessing, resampling, training multiple base models, and aggregating their predictions using a meta-classifier.


## Methodology

### 1. Boolean Conversion

Boolean columns in the dataset are converted to integers to ensure compatibility with the machine learning models, which require numerical input.

### 2. Feature Normalization

All features, except the target column, are scaled using Min-Max normalization. This transforms the values to a [0, 1] range, ensuring uniformity across all features and improving convergence for gradient-based models.

### 3. Handling Class Imbalance

The dataset may be imbalanced (e.g., more negatives than positives). To fix this, a combination technique called SMOTEENN is used. It balances the dataset by both oversampling the minority class (using SMOTE) and cleaning the noisy samples (using ENN - Edited Nearest Neighbors).

### 4. Train-Test Split

The balanced dataset is split into training and testing sets using stratified sampling. This ensures that the proportion of positive and negative cases remains consistent across both sets.

### 5. Base Models

Three strong classifiers are selected as base learners:
- XGBoost (eXtreme Gradient Boosting)
- LightGBM (Light Gradient Boosting Machine)
- Support Vector Classifier (SVC)

These models are trained in parallel during the stacking process.

### 6. Stacking Ensemble

The outputs (predictions) of the base models are combined using a meta-classifier, which in this case is Logistic Regression. The stacking classifier also has access to the original input features, which allows the meta-model to make more informed decisions.

5-fold cross-validation is used during training to improve generalization and reduce overfitting.

### 7. Evaluation


The model achieves strong performance:
- Accuracy: ~99.73%
- Precision: ~99.71%
- Recall: ~99.76%
- F1 Score: ~99.73%
- AUC: 1.00

![image](https://github.com/user-attachments/assets/f4266a25-199b-4ba2-8ea3-4a1bdce81bb0)


---

## Comparing Time Taken by Each Approach

| Approach              | Time Taken |
|-----------------------|------------|
| OUEFS + GA-HPO        | > 15 hrs   |
| Autoencoder + ANN     | 3 hrs      |
| TabNet                | 29 mins    |
| Stacking Ensemble     | 9 hrs      |

---

## Attempted Methods (But Failed)

| Method               | Accuracy | Precision (Class 1) | Recall (Class 1) | F1 Score (Class 1) | Conclusion |
|----------------------|----------|---------------------|------------------|---------------------|------------|
| **Gaussian Mixture Model (GMM)** | 0.6001   | 0.7468              | 0.3110           | 0.4391              | Poor recall means it missed many actual positives. Not ideal for imbalanced data. |
| **Local Outlier Factor**         | 0.9787   | 0.0390              | 0.0337           | 0.0362              | Extremely low precision & recall for positive class. Treats outliers poorly in high imbalance. |
| **Elliptic Envelope**           | 0.9780   | 0.0076              | 0.0066           | 0.0071              | Almost useless for detecting positive cases. Strong bias toward majority class. |
| **One-Class KMeans**            | 0.9445   | 0.0643              | 0.2703           | 0.1039              | Slightly better recall than others, but still far from usable. Weak at class separation. |
| **Isolation Forest**            | 0.9217   | 0.0436              | 0.2666           | 0.0750              | Better than Envelope/LOF, but poor class-1 performance. Outlier detection not suitable here. |

## Conclusion: Why These Methods Failed

1. **All these models are unsupervised or semi-supervised** (especially Isolation Forest, LOF, Elliptic Envelope, etc.), and they rely on anomaly or distribution assumptions. But your dataset is:
   - **Heavily imbalanced** (very few positives),
   - **Well-structured with labeled data** (which supervised methods can use better).

2. These models treat class-1 (positive COVID cases) as *outliers*, but real-world COVID prediction requires learning subtle patterns—*not just rarity*.

3. Outlier-based approaches **don’t generalize well when positive cases form patterns similar to negatives** in some features, especially after preprocessing like normalization and SMOTE.

4. Their **low recall** means they failed to catch actual positive cases, which is dangerous in a health prediction context.


