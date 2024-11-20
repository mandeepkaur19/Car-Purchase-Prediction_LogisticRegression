# Car-Purchase-Prediction_LogisticRegression
This repository contains code for predicting car purchases based on customer age and salary using a Logistic Regression model. The project includes data preprocessing, model training, evaluation, and making future predictions.
# Description of the Code
This code implements a Logistic Regression model to predict whether a customer will purchase a car based on their age and salary. The project involves data preprocessing, training a Logistic Regression classifier, evaluating model performance, and making future predictions.

* Importing Libraries:

  The code imports necessary libraries such as NumPy, Pandas, Matplotlib, and relevant modules from scikit-learn for data preprocessing, model building, and evaluation.


* Loading the Dataset:

  The dataset logit classification.csv is loaded using pd.read_csv(). It contains customer data including user ID, gender, age, salary, and whether they purchased a car (Yes/No). This dataset is assigned to the variable dataset.

* Feature Selection:

  The independent variables age and salary (columns 2 and 3) are selected and stored in X, while the dependent variable purchased (column 4) is stored in y.

* Splitting the Data:

  The data is split into training and testing sets using train_test_split() from scikit-learn. 25% of the data is reserved for testing, and the remaining 75% is used for training.

* Data Scaling:

  The StandardScaler is used to standardize the features by removing the mean and scaling to unit variance. Both the training and testing datasets are transformed using the fitted scaler.

* Logistic Regression Model:

  A Logistic Regression classifier is instantiated and trained on the scaled training data (X_train, y_train).
  The model then predicts the target variable (y_pred) using the test data (X_test).

* Evaluating the Model:

  The model’s performance is evaluated using several metrics:
  * Confusion Matrix: A matrix is printed to show the number of correct and incorrect predictions.
  * Accuracy Score: The accuracy of the model is calculated, showing how many predictions were correct out of the total.
  * Classification Report: This provides a detailed report with precision, recall, F1-score, and support for each class.
  * Training and Testing Score: The model’s training and testing performance (R² score) are printed to indicate bias and variance.

* Future Predictions:

A new dataset (final1.csv), containing age and salary of potential customers, is loaded. This dataset does not contain information on whether the customer has purchased a car.

This dataset is transformed and scaled just like the original data, and predictions are made for whether each customer will purchase a car.

The predictions are added as a new column (y_pred) to the d2 DataFrame, which is a copy of the original final1.csv dataset.

The new dataset with the predictions is saved as a new CSV file (pred_model.csv), which includes the predicted values for whether each customer will buy a car or not.
