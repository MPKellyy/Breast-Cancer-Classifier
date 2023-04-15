import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, recall_score, confusion_matrix, precision_score, f1_score, \
    accuracy_score
import json

# Global Variable(s)
fig_count = 1


# Function that trains classifier model and computes its accuracy
# Inputs:
# feature_matrix - numpy matrix of features
# solution_vector - numpy column vector of solutions
# test_size - decimal proportion of data used to train model
# model_type - string value of the model to train, defaults to logistic regression
# show_metrics - set True to view metrics in console
# Returns: classifier model, dictionary of model metrics
def create_classifier_model(feature_matrix, solution_vector, test_size, model_type, pos_class="R", show_metrics=False):
    metrics = {"model_type": model_type}
    valid_types = ["Logistic Regression", "SVM Classifier", "MLP Classifier"]

    assert model_type in valid_types, "Specified model type not valid"

    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(feature_matrix, solution_vector, test_size=test_size)

    # Creating model
    if model_type == "SVM Classifier":
        model = SVC()
    elif model_type == "MLP Classifier":
        model = MLPClassifier(hidden_layer_sizes=(100,), activation="tanh", alpha=0.001, solver="adam",
                              early_stopping=True)
    else:
        model = LogisticRegression(max_iter=10000)

    # model.fit(feature_matrix, solution_vector)
    model.fit(x_train, y_train)

    # Predicting with test features
    y_pred = model.predict(x_test)

    # Computing the recall score
    sensitivity = recall_score(y_test, y_pred, pos_label=pos_class)
    metrics["recall"] = sensitivity

    # Computing specificity
    confusion = confusion_matrix(y_test, y_pred)
    true_neg, false_pos, false_neg, true_pos = confusion.ravel()
    specificity = true_neg / (true_neg + false_pos)
    metrics["specificity"] = specificity

    # Computing precision
    precision = precision_score(y_test, y_pred, pos_label=pos_class, zero_division=1)
    metrics["precision"] = precision

    # Computing f1 score
    f1 = f1_score(y_test, y_pred, pos_label=pos_class)
    metrics["f1_score"] = f1

    # Computing accuracy
    accuracy = accuracy_score(y_test, y_pred)
    metrics["accuracy"] = accuracy

    # Computing cross validation on testing data
    test_cross_scores = cross_val_score(model, x_test, y_test, cv=5)
    test_cross_scores = test_cross_scores.tolist()
    metrics["test_cross_validation_scores"] = test_cross_scores

    # Computing cross validation on input data
    input_cross_scores = cross_val_score(model, feature_matrix, solution_vector, cv=5)
    input_cross_scores = input_cross_scores.tolist()
    metrics["input_cross_validation_scores"] = input_cross_scores

    # Displaying metrics to console if specified
    if show_metrics:
        print("Model type: " + model_type)
        print("Sensitivity (recall): " + str(sensitivity))
        print("Specificity: " + str(specificity))
        print("Precision: " + str(precision))
        print("F1 Score: " + str(f1))
        print("Accuracy: " + str(accuracy))
        print("Test Cross Validation Scores: " + ", ".join([str(score) for score in test_cross_scores]))
        print("Input Cross Validation Scores: " + ", ".join([str(score) for score in input_cross_scores]))

    # Plotting training vs validation
    plot_training_validation_error(model_type + " Classification Test", "Recurrence Outcomes", y_test, y_pred)

    # Returning model and computed scores
    return model, metrics


# Function that creates regression model and computes RMSE
# Can print predictions based on input data by setting show_metrics to True
# Inputs:
# feature_matrix - numpy matrix of features
# solution_vector - numpy column vector of solutions
# test_size - decimal proportion of data used to train model
# model_type - string value of the model to train, defaults to linear regression
# show_metrics - set True to view metrics in console
# Returns: regression model, dictionary of model metrics
def create_regression_model(feature_matrix, solution_vector, test_size, model_type="Linear Regression", show_metrics=False):
    metrics = {"model_type": model_type}
    valid_types = ["Linear Regression", "MLP Regressor", "SVM Regression"]

    assert model_type in valid_types, "Specified model type not valid"

    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(feature_matrix, solution_vector, test_size=test_size)

    if model_type == "MLP Regressor":
        model = MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=10000, alpha=0.001, activation="relu",
                             solver="lbfgs")  # relu - lbfgs
        title1 = "MLP Regressor Recurrence Predictions in Months"
        title2 = "MLP Regressor Recurrence Predictions in Years"
    elif model_type == "SVM Regression":
        model = SVR(kernel="rbf")
        title1 = "SVM Regression Recurrence Predictions in Months"
        title2 = "SVM Regression Recurrence Predictions in Years"
    else:
        # Create logistic regression model
        model = Lasso(alpha=0.7)
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.fit_transform(x_test)
        title1 = "Linear Regression Recurrence Predictions in Months"
        title2 = "Linear Regression Recurrence Predictions in Years"

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    # y_pred[y_pred < 0] = 0
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    metrics["RMSE"] = rmse

    # If user wants to see predictions on input data, print to console
    if show_metrics:
        print("Model type: " + model_type)
        print("RMSE: " + str(rmse))

        prediction_log = ["Comparing model predictions to actual values:"]

        print("Comparing model predictions to actual values:")

        for prediction, time in zip(y_pred, y_test):
            prediction_str = "Prediction: " + str(prediction) + "\t\tActual: " + str(time)
            prediction_log.append(prediction_str)
            print(prediction_str)

        metrics["Predictions"] = prediction_log

    # Plotting training vs validation
    plot_training_validation_error(title1, "Months", y_test, y_pred)
    plot_training_validation_error(title2, "Years", np.floor_divide(y_test, 12), np.floor_divide(y_pred, 12))

    return model, metrics


# Visualizes the predictions after training vs actual values (validation step)
# Saves a png figure of validation plot
# Inputs:
# fig_title - string title of figure
# y_label - string label of y-axis
# y_actual - test/actual y values (control)
# y_predictions - predicted y values (dependent variables)
def plot_training_validation_error(fig_title, y_label, y_actual, y_predictions):
    global fig_count

    sample_range = list(range(1, y_actual.shape[0] + 1))
    plt.title(fig_title)
    plt.ylabel(y_label)
    plt.xlabel("Test Sample #")
    plt.scatter(sample_range, y_actual, marker="+", label="actual")
    plt.scatter(sample_range, y_predictions, marker="x", label="prediction")

    plt.xticks(sample_range, rotation=90, fontsize=6)
    plt.legend(loc="best")

    # Save figure as png
    plt.savefig("Figure" + str(fig_count) + ".png", dpi=200)
    plt.close()

    fig_count += 1


# Linear regression part #####################
# Loading in dataset
wpbc_df = pd.read_csv("wpbc.data", header=None)

# We are only tracking recurrent cancer

# Removing non-recurrent data
recur_df = wpbc_df[~(wpbc_df.iloc[:, :] == 'N').any(axis=1)].reset_index(drop=True)

# Removing samples with missing features
recur_df = recur_df[~(recur_df.iloc[:, :] == '?').any(axis=1)].reset_index(drop=True)

# Creating feature matrix
feature_matrix = recur_df.drop(recur_df.columns[[0, 1, 2]], axis=1)  # Dropping id, label, time
feature_matrix = feature_matrix.apply(pd.to_numeric, errors='ignore')
feature_matrix = feature_matrix.to_numpy()

# Creating solution vector
solution_vector = recur_df.iloc[:, 2]  # Only saving time column
solution_vector = solution_vector.to_numpy()

# Creating linear regression model while printing out prediction comparisons
lin_reg_model, lin_reg_metrics = create_regression_model(feature_matrix, solution_vector, 0.3,
                                                     model_type="Linear Regression", show_metrics=True)
print("\n")

mlp_reg_model, mlp_reg_metrics = create_regression_model(feature_matrix, solution_vector, 0.3,
                                                         model_type="MLP Regressor", show_metrics=True)
print("\n")

svm_reg_model, svm_reg_metrics = create_regression_model(feature_matrix, solution_vector, 0.3,
                                                         model_type="SVM Regression", show_metrics=True)
print("\n")


# Classification part #####################
# We are only classifying recur/non-recur

# Dropping id values and time values (col 0, 2), they will skew results (time is diff between classes)
recur_nonrecur_df = wpbc_df.drop(wpbc_df.columns[[0, 2]], axis=1)

# Sample containing '?' feature values are currently removed, however, this may affect predictions
recur_nonrecur_df = recur_nonrecur_df[~(recur_nonrecur_df.iloc[:, :] == '?').any(axis=1)].reset_index(drop=True)

# Creating feature matrix
feature_matrix = recur_nonrecur_df.drop(recur_nonrecur_df.columns[0], axis=1)  # Dropping labels
feature_matrix = feature_matrix.to_numpy()
solution_vector = recur_nonrecur_df.iloc[:, 0]  # Saving only the labels
solution_vector = solution_vector.to_numpy()

print("Predicting whether or not a tumor is recur/non-recur using wpbc.data")

# Using logistic regression
print("Logistic Regression")
log_model, log_class_metrics = create_classifier_model(feature_matrix, solution_vector, 0.3, "Logistic Regression",
                                                 pos_class="R", show_metrics=True)
print("\n")

# Using neural net
print("MLP Classifier")
mlp, mlp_class_metrics = create_classifier_model(feature_matrix, solution_vector, 0.3, "MLP Classifier", pos_class="R",
                                                 show_metrics=True)
print("\n")

# Using SVM
print("SVM")
svm, svm_class_metrics = create_classifier_model(feature_matrix, solution_vector, 0.3, "SVM Classifier", pos_class="R",
                                           show_metrics=True)
print("\n")

# Exporting all metrics to json
all_metrics = [lin_reg_metrics, mlp_reg_metrics, svm_reg_metrics, log_class_metrics, mlp_class_metrics, svm_class_metrics]

with open("metrics.json", "w") as file:
    json.dump(all_metrics, file, indent=4)
#########################################################
