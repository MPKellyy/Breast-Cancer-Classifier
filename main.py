import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score


# Function that trains classifier model and computes its accuracy
# Inputs:
# feature_matrix - numpy matrix of features
# solution_vector - numpy column vector of solutions
# test_size - decimal proportion of data used to train model
# model_type - string value of the model to train, defaults to logistic regression
# Returns: classifier model, decimal value of accuracy as a percentage
def create_classifier_model(feature_matrix, solution_vector, test_size, model_type):
    valid_types = ["logistic", "svm", "mlp"]

    assert model_type in valid_types, "Specified model type not valid," \
                                      " choose one from the following: 'logistic', 'svm', 'mlp'"

    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(feature_matrix, solution_vector, test_size=test_size)

    # Creating model
    if model_type == "svm":
        model = SVC()
    elif model_type == "mlp":
        model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=4000, activation="tanh", alpha=0.001, solver="adam")
    else:
        model = LogisticRegression(max_iter=10000)

    # model.fit(feature_matrix, solution_vector)
    model.fit(x_train, y_train)

    # Predicting with test features
    y_pred = model.predict(x_test)

    # Returning computed scores
    return model, cross_val_score(model, x_test, y_test, cv=4), cross_val_score(model, feature_matrix, solution_vector, cv=4)


# Function that creates lasso model and computes RMSE
# Can print predictions based on input data by setting compare_predictions to True
# Inputs:
# feature_matrix - numpy matrix of features
# solution_vector - numpy column vector of solutions
# Returns: lasso model, RMSE value
def create_lasso_model(feature_matrix, solution_vector, test_size, compare_predictions=False):
    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(feature_matrix, solution_vector, test_size=test_size)

    # Create logistic regression model
    model = Lasso(alpha=0.7)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.fit_transform(x_test)

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    # y_pred[y_pred < 0] = 0
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    # If user wants to see predictions on input data, print to console
    if compare_predictions:
        print("Comparing linear regression model predictions to actual values:")

        for prediction, time in zip(y_pred, y_test):
            print("Prediction: " + str(prediction) + "\t\tActual: " + str(time))

    return model, rmse


# Function that creates mlp regressor model and computes RMSE
# Can print predictions based on input data by setting compare_predictions to True
# Inputs:
# feature_matrix - numpy matrix of features
# solution_vector - numpy column vector of solutions
# Returns: mlp regressor model, RMSE value
def create_mlp_regression_model(feature_matrix, solution_vector, test_size, compare_predictions=False):
    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(feature_matrix, solution_vector, test_size=test_size)

    # Create mlp regressor model
    model = MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=10000, alpha=0.001, activation="relu", solver="lbfgs") # relu - lbfgs
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    # y_pred[y_pred < 0] = 0
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    # If user wants to see predictions on input data, print to console
    if compare_predictions:
        print("Comparing neural net regression predictions to actual values:")

        for prediction, time in zip(y_pred, y_test):
            print("Prediction: " + str(prediction) + "\t\tActual: " + str(time))

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
lin_reg_model, lin_reg_rmse = create_lasso_model(feature_matrix, solution_vector, 0.3, compare_predictions=True)
print("Linear Regression Model RMSE: " + str(lin_reg_rmse))
print("\n")

mlp_reg_model, mlp_reg_rmse = create_mlp_regression_model(feature_matrix, solution_vector, 0.3, compare_predictions=True)
print("MLP Regression Model RMSE: " + str(mlp_reg_rmse))
print("\n")

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
recur_nonrecur_log_model, log_test_cv_scores, log_actual_cv_scores = create_classifier_model(feature_matrix, solution_vector, 0.3, "logistic")
print("Logistic Regression Classification Test Cross Validation Scores: " + str(log_test_cv_scores))
print("Logistic Regression Classification Actual Cross Validation Scores: " + str(log_actual_cv_scores))
print("\n")

# Using neural net
mlp, mlp_test_cv_scores, mlp_actual_cv_scores = create_classifier_model(feature_matrix, solution_vector, 0.1, "mlp")
print("Neural Net Classification Test Cross Validation Scores: ", str(mlp_test_cv_scores))
print("Neural Net Classification Actual Cross Validation Scores: ", str(mlp_actual_cv_scores))
print("\n")

# Using SVM
svm, svm_test_cv_scores, svm_actual_cv_scores = create_classifier_model(feature_matrix, solution_vector, 0.3, "svm")
print("SVM Classification Test Cross Validation Scores: ", str(svm_test_cv_scores))
print("SVM Classification Cross Actual Cross Validation Scores: ", str(svm_actual_cv_scores))
print("\n")
#########################################################

