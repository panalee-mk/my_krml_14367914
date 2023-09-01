# Solution:
def print_regressor_scores(y_preds, y_actuals, set_name=None):
    """Print the RMSE and MAE for the provided data

    Parameters
    ----------
    y_preds : Numpy Array
        Predicted target
    y_actuals : Numpy Array
        Actual target
    set_name : str
        Name of the set to be printed

    Returns
    -------
    """
    from sklearn.metrics import mean_squared_error as mse
    from sklearn.metrics import mean_absolute_error as mae

    print(f"RMSE {set_name}: {mse(y_actuals, y_preds, squared=False)}")
    print(f"MAE {set_name}: {mae(y_actuals, y_preds)}")

# Solution:
def assess_regressor_set(model, features, target, set_name=''):
    """Save the predictions from a trained model on a given set and print its RMSE and MAE scores

    Parameters
    ----------
    model: sklearn.base.BaseEstimator
        Trained Sklearn model with set hyperparameters
    features : Numpy Array
        Features
    target : Numpy Array
        Target variable
    set_name : str
        Name of the set to be printed

    Returns
    -------
    """
    preds = model.predict(features)
    print_regressor_scores(y_preds=preds, y_actuals=target, set_name=set_name)


# Solution:
def fit_assess_regressor(model, X_train, y_train, X_val, y_val):
    """Train a regressor model, print its RMSE and MAE scores on the training and validation set and return the trained model

    Parameters
    ----------
    model: sklearn.base.BaseEstimator
        Instantiated Sklearn model with set hyperparameters
    X_train : Numpy Array
        Features for the training set
    y_train : Numpy Array
        Target for the training set
    X_train : Numpy Array
        Features for the validation set
    y_train : Numpy Array
        Target for the validation set

    Returns
    sklearn.base.BaseEstimator
        Trained model
    -------
    """
    model.fit(X_train, y_train)
    assess_regressor_set(model, X_train, y_train, set_name='Training')
    assess_regressor_set(model, X_val, y_val, set_name='Validation')
    return model



def print_classifier_scores(y_probs, y_actuals, set_name=None):
    """Print the AUC-ROC score for the provided data

    Parameters
    ----------
    y_probs : Numpy Array
        Predicted probabilities of positive class
    y_actuals : Numpy Array
        Actual binary labels (0 or 1)
    set_name : str
        Name of the set to be printed

    Returns
    -------
    """
    from sklearn.metrics import roc_auc_score

    auc_roc = roc_auc_score(y_actuals, y_probs)
    print(f"AUC-ROC {set_name}: {auc_roc}")

def assess_classifier_set(model, features, target, set_name=''):
    """Save the predicted probabilities from a trained classification model on a given set and print its AUC-ROC score

    Parameters
    ----------
    model: sklearn.base.ClassifierMixin
        Trained Sklearn classification model with set hyperparameters
    features : Numpy Array
        Features
    target : Numpy Array
        Actual binary labels (0 or 1)
    set_name : str
        Name of the set to be printed

    Returns
    -------
    """
    probs = model.predict_proba(features)[:, 1]  # Probabilities of the positive class
    auc_roc = roc_auc_score(target, probs)
    print(f"AUC-ROC {set_name}: {auc_roc}")


def fit_assess_classifier(model, X_train, y_train, X_val, y_val):
    """Train a classifier model, print its AUC-ROC scores on the training and validation set, and return the trained model

    Parameters
    ----------
    model: sklearn.base.ClassifierMixin
        Instantiated Sklearn classification model with set hyperparameters
    X_train : Numpy Array
        Features for the training set
    y_train : Numpy Array
        Target for the training set
    X_val : Numpy Array
        Features for the validation set
    y_val : Numpy Array
        Target for the validation set

    Returns
    -------
    sklearn.base.ClassifierMixin
        Trained model
    """
    model.fit(X_train, y_train)
    assess_classifier_set(model, X_train, y_train, set_name='Training')
    assess_classifier_set(model, X_val, y_val, set_name='Validation')
    return model