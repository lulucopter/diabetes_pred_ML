# SGH Technical Assessment: Diabetes Prediction on NHANES Dataset

# Notebook Contents

## Environment set up:
1. Download dataset
2. install relevant ML libraries

## Data Preprocessing
1. Ensure data consistency with data repo details
2. Impute Invalid and Missing values with relevant values for ML training

## Database Integration
1. Set up and save data in SQLite3 database
2. Brief data exploration using SQL Queries

## Exploratory Data Analysis
1. Correlation Heatmap: Identify potential feature importance to diabetes prediction
2. Age Distribution Histogram: Diabetes patients tend to have higher mean age
3. BMI Distribution Histogram: Diabetes patients tend to have higher mean BMI
4. Weight Distribution Histogram: Diabetes patients tend to have slightly higher mean weight
5. Waist Circumference Histogram: Diabetes patients tend to have slightly higher mean waist size
6. Albumin Levels Histogram: Diabetes patients tend to have very slightly lower mean albumin levels
7. Serum Creatinine Distribution Histogram: SCr levels don't seem to have an effect
8. Family Income Distribution Histogram: Income don't seem to affect diabetes as much
9. Race/Ethnicity Distribution Histogram: Non-Hispanic Whites seem to have a big % difference when comparing diabetes rate in this RE group

## Feature Engineering
Engineering likely features based on existing literature on diabetes prevalance
1. Binning of features:
   - Age Group
   - BMI Category
2. Waist to Height Ratio
3. Interaction Terms:
   - Age & BMI Interaction
   - Age & Waist-to-height-ratio Interaction

## Training, Testing and Evaluation of Models
1. Prepare train test split (80/20)
2. Remove features that directly gives away diabetes status. Eg:
   -  gh levels
   -  tx: On treatment using Insulin/Diabetes Meds
   -  dx: Diagnosed with Diabetes or Pre-Diabetes

### Initial ML Training: Logistic Regression
1. Normal LogReg model: Acc: 0.90, Pr: 0.43, Rec: 0.07, ROC-AUC: 0.53
2. Feature Selection: RFECV LogReg model: Acc: 0.90, Pr: 0.00, Rec: 0.00, ROC-AUC: 0.50
   - These low precision hints to an imbalanced dataset. Which is true, as we can see that Num Diabetes is about 10x lower than those without

### Handle imbalanced dataset with SMOTE (Synthetic Minority Over-sampling Technique)
1. Use SMOTE to get new train and test datasets

### Logistic Regression
1. Normal LogReg Model: Acc: 0.91, Pr: 0.91, Rec: 0.90, ROC-AUC: 0.91
   - Decent Performance, but can be better. LR is too naive

### Decision Tree
1. Normal DT Model: Acc: 0.90, Pr: 0.87, Rec: 0.92, ROC-AUC: 0.90
   - Room to improve with hyper parameter tuning (Fine tuning)
2. Finetuned DT Model: Acc: 0.90, Pr: 0.90, Rec: 0.90, ROC-AUC: 0.90
   - Better precision, but ROC and Acc doesn't change. DT may still be too naive for a ML model here
  
### Random Forest
1. Normal RF Model: Acc: 0.94, Pr: 0.94, Rec: 0.94, ROC-AUC: 0.94
   - Straight away see significant improvements across all 4 metrics. Let's finetune and improve this model
2. Finetuned RF Model: Acc: 0.95, Pr: 0.95, Rec: 0.94, ROC-AUC: 0.95
   - Very good model performance (>= 95%) !!

### Model comparison and selection
1. RF model performs the best across all boards. Even without finetuning. Let's use this RF model for validation and interpretation

## Model Validation
1. Doing a simply 5-fold Cross Validation, we get the scores: 0.82, 0.97, 0.96, 0.96, 0.97
2. Let's investigate the outlying first fold.
3. Manual 5 fold doesn't reproduce the outlying score: 0.94, 0.94, 0.94, 0.95, 0.95
4. Taken as the first CV was unlucky

## Model Interpretation
1. Feature Importance Graph
2. SHAP (SHapley Additive exPlanations)
3. Using both graphs, we can deduce the following important features in diabetes prediction
   - Age & Waist to Height Ratio Interaction
   - Age & BMI Interaction
   - Age
   - Waist Size
   - Waist to Height Ratio
   - Race/Ethnicity: Being/Not Being a Non-Hispanic-White
