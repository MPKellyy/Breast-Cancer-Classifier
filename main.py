import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression


# Function that computes creates logistic regression model and its accuracy
# Inputs:
# feature_matrix - numpy matrix of features
# solution_vector - numpy column vector of solutions
# test_size - decimal proportion of data used to train model
# Returns: logistic regression model, decimal value of accuracy as a percentage
def create_logistic_regression_model(feature_matrix, solution_vector, test_size):
    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(feature_matrix, solution_vector, test_size=test_size)

    # Create logistic regression model
    model = LogisticRegression(max_iter=3000)
    model.fit(feature_matrix, solution_vector)

    # Predicting with test features
    y_pred = model.predict(x_test)

    # Returning computed score
    return model, accuracy_score(y_test, y_pred)


# Function that computes creates linear regression model
# Can print predictions based on input data by setting compare_predictions to True
# Inputs:
# feature_matrix - numpy matrix of features
# solution_vector - numpy column vector of solutions
# Returns: linear regression model
def create_linear_regression_model(feature_matrix, solution_vector, compare_predictions=False):
    # Create logistic regression model
    model = LinearRegression()
    model.fit(feature_matrix, solution_vector)

    # If user wants to see predictions on input data, print to console
    if compare_predictions:
        print("Comparing linear regression model predictions to actual values:")

        for feature, time in zip(feature_matrix, solution_vector):
            prediction = model.predict([feature.tolist()])[0]
            print("Prediction: " + str(prediction) + "\t\tActual: " + str(time))
        print("\n")

    return model


# This is reccur/non-recurr dataset #####################
# May not want to use this for classification...
# This should use linear regression as it relates to time

# Loading in dataset
wpbc_df = pd.read_csv("wpbc.data", header=None)

# Performing linear regression on recurrence class
# We are only required to track one class over time, prof suggested recurrence
# Removing non-recurrent data
recur_df = wpbc_df[~(wpbc_df.iloc[:, :] == 'N').any(axis=1)].reset_index(drop=True)
# Removing samples with missing features
# TODO: sample containing '?' feature values are currently removed, however, this may affect predictions
recur_df = recur_df[~(recur_df.iloc[:, :] == '?').any(axis=1)].reset_index(drop=True)
# Creating feature matrix
feature_matrix = recur_df.drop(recur_df.columns[[0, 1, 2]], axis=1)  # Dropping id, label, time
feature_matrix = feature_matrix.apply(pd.to_numeric, errors='ignore')
feature_matrix = feature_matrix.to_numpy()
# Creating solution vector
solution_vector = recur_df.iloc[:, 2]  # Only saving time column
solution_vector = solution_vector.to_numpy()
# Creating linear regression model while printing out prediction comparisons
create_linear_regression_model(feature_matrix, solution_vector, compare_predictions=True)


# In case prof wants to classify recur/non-recur, I included the code below
# TODO: Currently dropping id values and time values (col 0, 2), they will skew results (time is diff between classes)
recur_nonrecur_df = wpbc_df.drop(wpbc_df.columns[[0, 2]], axis=1)
# TODO: sample containing '?' feature values are currently removed, however, this may affect predictions
recur_nonrecur_df = recur_nonrecur_df[~(recur_nonrecur_df.iloc[:, :] == '?').any(axis=1)].reset_index(drop=True)
# recur_nonrecur_df = recur_nonrecur_df.replace('?', 0)
# Creating feature matrix
feature_matrix = recur_nonrecur_df.drop(recur_nonrecur_df.columns[0], axis=1)  # Dropping labels
feature_matrix = feature_matrix.to_numpy()
solution_vector = recur_nonrecur_df.iloc[:, 0]  # Saving only the labels
solution_vector = solution_vector.to_numpy()
print("Predicting whether or not a tumor is recurr/non-recurr using wpbc.data")
recur_nonrecur_log_model, accuracy = create_logistic_regression_model(feature_matrix, solution_vector, 0.30)
print("wpbc.data Logistic Regression Accuracy: " + str(accuracy))
print("\n")
#########################################################


# Part of malignant/benign data #########################
# We'll need this for the classification part of project

# Loading in data
wdbc_df = pd.read_csv("wdbc.data", header=None)
# Data filtering
wdbc_class_df = wdbc_df.drop(wdbc_df.columns[[0]], axis=1)  # Dropping ids
# Creating feature matrix
feature_matrix = wdbc_class_df.drop(wdbc_class_df.columns[0], axis=1)  # Dropping labels
feature_matrix = feature_matrix.to_numpy()
# Creating solution vector
solution_vector = wdbc_class_df.iloc[:, 0]  # Keeping only labels
solution_vector = solution_vector.to_numpy()
print("Predicting whether or not a tumor is benign/malignant using wdbc.data")
wdbc_log_model, accuracy = create_logistic_regression_model(feature_matrix, solution_vector, 0.30)
print("wdbc.data Logistic Regression Accuracy: " + str(accuracy))
print("\n")
#########################################################


# Another part of malignant/benign data #################
# We'll also need this for the classification part of project

# Loading in data
wis_df = pd.read_csv("breast-cancer-wisconsin.data", header=None)
# Data filtering
wis_class_df = wis_df.drop(wis_df.columns[[0]], axis=1)  # Dropping id column
# TODO: sample containing '?' feature values are currently removed, however, this may affect predictions
wis_class_df = wis_class_df[~(wis_class_df.iloc[:, :] == '?').any(axis=1)].reset_index(drop=True)
# Creating feature matrix
feature_matrix = wis_class_df.drop(wis_class_df.columns[9], axis=1)  # Dropping labels
feature_matrix = feature_matrix.to_numpy()
# Creating solution vector
solution_vector = wis_class_df.iloc[:, 9]  # Saving only labels (three labels refer to breast-cancer-wisconsin.names)
solution_vector = solution_vector.to_numpy()
print("Predicting whether or not a tumor is benign/malignant using breast-cancer-wisconsin.data")
wis_log_model, accuracy = create_logistic_regression_model(feature_matrix, solution_vector, 0.30)
print("breast-cancer-wisconsin.data Logistic Regression Accuracy: " + str(accuracy))
print("\n")
#########################################################

