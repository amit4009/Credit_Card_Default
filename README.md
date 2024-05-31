# Credit_Card_Default

This research employed a binary variable, default payment (Yes = 1, No = 0), as the response variable. This study reviewed the literature and used the following 23 variables as explanatory variables: X1: Amount of the given credit (NT dollar): it includes both the individual consumer credit and his/her family (supplementary) credit. X2: Gender (1 = male; 2 = female). X3: Education (1 = graduate school; 2 = university; 3 = high school; 4 = others). X4: Marital status (1 = married; 2 = single; 3 = others). X5: Age (year). X6 - X11: History of past payment. We tracked the past monthly payment records (from April to September, 2005) as follows: X6 = the repayment status in September, 2005; X7 = the repayment status in August, 2005; . . .;X11 = the repayment status in April, 2005. The measurement scale for the repayment status is: -1 = pay duly; 1 = payment delay for one month; 2 = payment delay for two months; . . .; 8 = payment delay for eight months; 9 = payment delay for nine months and above. X12-X17: Amount of bill statement (NT dollar). X12 = amount of bill statement in September, 2005; X13 = amount of bill statement in August, 2005; . . .; X17 = amount of bill statement in April, 2005. X18-X23: Amount of previous payment (NT dollar). X18 = amount paid in September, 2005; X19 = amount paid in August, 2005; . . .;X23 = amount paid in April, 2005.

# Description
This project aims to predict whether a customer will default on their credit card payment next month. Using a variety of customer attributes, the model can help financial institutions manage risk and make informed credit decisions. The dataset used is sourced from the Taiwan Credit Crisis.

# Key Features:
Data Preprocessing: Cleaning and preparing the dataset for modeling.<br>
Feature Engineering: Creating new features to improve model performance.<br>
Model Selection: Comparing different machine learning models to find the best one.<br>
Hyperparameter Tuning: Optimizing the model for better accuracy.<br>
Deployment: Using Flask to deploy the model as a web application.<br>

# Algorithm
1. Data Preprocessing<br>
Handling Missing Values: Checked for and imputed missing values.<br>
Outlier Detection: Identified and treated outliers in numerical features.<br>
Encoding Categorical Variables: Converted categorical variables into numerical format using techniques like one-hot encoding.<br>
Feature Scaling: Standardized numerical features using StandardScaler.<br>
2. Exploratory Data Analysis (EDA)<br>
Data Visualization: Used histograms, box plots, and correlation matrices to understand data distributions and relationships.<br>
Correlation Analysis: Identified the relationships between features and the target variable.<br>
3. Feature Engineering<br>
Creation of New Features: Developed new features such as Average Bill Amount, Payment Ratio, and Payment Delay to capture more information from the data.<br>
Dimensionality Reduction: Used techniques like PCA to reduce the number of features while retaining most of the information.<br>
4. Model Building<br>
Model Training: Trained several machine learning models, including:<br>
Logistic Regression<br>
Random Forest<br>
XGBoost<br>
Support Vector Machines (SVM)<br>

Model Evaluation: Evaluated models using accuracy, precision, recall, F1-score, and ROC-AUC.

5. Hyperparameter Tuning<br>
Grid Search: Conducted exhaustive search over specified parameter values for an estimator.<br>
Random Search: Sampled a fixed number of parameter settings from a specified distribution.<br>
6. Model Deployment<br>
Flask Application: Deployed the final model using Flask to provide real-time predictions via a web interface.<br>

# Results
The final model achieved the following metrics:<br>
Accuracy: 82%<br>
Precision: 78%<br>
Recall: 70%<br>
F1-Score: 74%<br>

The model effectively reduced false positives by 15%, enhancing the risk assessment capabilities.

# The use app image.
![alt text](image.png)

# Observations
Feature Importance: Payment history and bill amounts were significant predictors of default, while demographic features like age and education level had less impact.<br>
Model Performance: Ensemble models such as Random Forest and XGBoost outperformed simpler models like Logistic Regression and Decision Trees in terms of accuracy and robustness.<br>
Data Imbalance: Addressing the imbalance in the target variable using techniques such as SMOTE (Synthetic Minority Over-sampling Technique) improved the model's performance.<br>
Hyperparameter Tuning: Fine-tuning model parameters significantly improved performance metrics, demonstrating the importance of this step.<br>

# Conclusion
The credit card default prediction model developed in this project provides a reliable tool for assessing credit risk. By identifying key predictors of default, the model helps financial institutions make data-driven decisions to minimize risk. Future enhancements could include incorporating more advanced machine learning techniques and expanding the dataset to improve generalizability.










