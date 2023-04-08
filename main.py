import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor


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


# Function that computes creates mlp classifier model and its accuracy
# Inputs:
# feature_matrix - numpy matrix of features
# solution_vector - numpy column vector of solutions
# test_size - decimal proportion of data used to train model
# Returns: mlp classifier model, decimal value of accuracy as a percentage
def create_mlp_classifier_model(feature_matrix, solution_vector, test_size):
    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(feature_matrix, solution_vector, test_size=test_size)

    model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=4000, activation="tanh", alpha=0.001, solver="adam")
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    # Returning computed score
    return model, accuracy_score(y_test, y_pred)


# Function that creates linear regression model and computes RMSE
# Can print predictions based on input data by setting compare_predictions to True
# Inputs:
# feature_matrix - numpy matrix of features
# solution_vector - numpy column vector of solutions
# Returns: linear regression model, RMSE value
def create_linear_regression_model(feature_matrix, solution_vector, compare_predictions=False):
    # Create logistic regression model
    model = LinearRegression()
    model.fit(feature_matrix, solution_vector)
    rmse = mean_squared_error(solution_vector, model.predict(feature_matrix), squared=False)

    # If user wants to see predictions on input data, print to console
    if compare_predictions:
        print("Comparing linear regression model predictions to actual values:")

        for feature, time in zip(feature_matrix, solution_vector):
            prediction = model.predict([feature.tolist()])[0]
            print("Prediction: " + str(prediction) + "\t\tActual: " + str(time))

    print("Model RMSE: " + str(rmse))
    print("\n")

    return model, rmse


# Function that creates mlp regressor model and computes RMSE
# Can print predictions based on input data by setting compare_predictions to True
# Inputs:
# feature_matrix - numpy matrix of features
# solution_vector - numpy column vector of solutions
# Returns: mlp regressor model, RMSE value
def create_mlp_regression_model(feature_matrix, solution_vector, compare_predictions=False):
    # Create mlp regressor model
    model = MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=10000, alpha=0.001, activation="logistic", solver="lbfgs")
    model.fit(feature_matrix, solution_vector)
    rmse = mean_squared_error(solution_vector, model.predict(feature_matrix), squared=False)

    # If user wants to see predictions on input data, print to console
    if compare_predictions:
        print("Comparing neural net regression predictions to actual values:")

        for feature, time in zip(feature_matrix, solution_vector):
            prediction = model.predict([feature.tolist()])[0]
            print("Prediction: " + str(prediction) + "\t\tActual: " + str(time))

    print("Model RMSE: " + str(rmse))
    print("\n")

    return model, rmse


# Linear regression part #####################
# Loading in dataset
wpbc_df = pd.read_csv("wpbc.data", header=None)

# We are only tracking recurrent cancer

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
lin_reg_model, lin_reg_rmse = create_linear_regression_model(feature_matrix, solution_vector, compare_predictions=True)
mlp_reg_model, mlp_reg_rmse = create_mlp_regression_model(feature_matrix, solution_vector, compare_predictions=True)

# Classification part #####################
# We are only classifying recur/non-recur

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

print("Predicting whether or not a tumor is recur/non-recur using wpbc.data")

# Using logistic regression
recur_nonrecur_log_model, accuracy = create_logistic_regression_model(feature_matrix, solution_vector, 0.30)
print("Logistic Regression Classification Accuracy: " + str(accuracy))

# Using neural net
mlp, mlp_accuracy = create_mlp_classifier_model(feature_matrix, solution_vector, 0.30)
print("Neural Net Classification Accuracy: ", str(mlp_accuracy))
#########################################################

