__author__ = 'dylanjf'

import os
import pickle
import logging
import scipy as sp
import numpy as np
from re import sub
from sklearn.grid_search import GridSearchCV
from sklearn import cross_validation

logger = logging.getLogger(__name__)

N_TREES = 300

INITIAL_PARAMS = {
    'LogisticRegression': {'C': 1, 'penalty': 'l2', 'class_weight': None},
    'RandomForestClassifier': {
        'n_estimators': N_TREES, 'n_jobs': 4,
        'min_samples_leaf': 2, 'bootstrap': False,
        'max_depth': 30, 'min_samples_split': 5, 'max_features': .1
    },
    'GradientBoostingClassifier': {
        'n_estimators': N_TREES, 'learning_rate': .08, 'max_features': 4,
        'min_samples_leaf': 2, 'min_samples_split': 5, 'max_depth': 7
    }
}

PARAM_GRID = {
    'LogisticRegression': {'C': [1, 2, 3, 5, 10, 100],
                           'penalty': ['l1', 'l2']},
    'ElasticNet': {'alpha': [.1, 1.0],
                   'l1_ratio': [0, .05, .5, .95, 1.0]},
    'GradientBoostingClassifier': {
        'learning_rate': [.01, .03, .05, .08], 'max_depth': [3, 4, 7], 'max_features': [4, 8, 12]
    }
}


class EnsembleGeneralization(object):
    """
    Implement stacking to combine several models.
    The base (stage 0) models can be either combined through
    simple averaging (fastest), or combined using a stage 1 generalizer
    (requires computing CV predictions on the train set).

    See http://ijcai.org/Past%20Proceedings/IJCAI-97-VOL2/PDF/011.pdf:
    "Stacked generalization: when does it work?", Ting and Witten, 1997

    Expects models to be a list of (model, dataset) tuples.
    If model selection is true, use grid search on globally defined hyperparams
    to get best feature combination
    """

    def __init__(self, models, score_func, stack, isContinuous=True,
                 for_model_select=False, pickle_root_dir=None):
        self.models = models
        self.for_model_select = for_model_select
        self.stack = stack
        self.score_func = score_func
        self.generalizer = MLR()
        self.isContinuous = isContinuous
        self.pickle_root_dir = pickle_root_dir

    def _combine_predictions(self, X_train, X_cv, y):
        """
        helper function for CV loop.

        assuming CV predictions are aligned for each model,
        return both the mean of the 3 models (for if stack = False)
        and the result of the Non Negative Least Squares.
        """
        if self.isContinuous:
            mean_preds = np.mean(X_cv, axis=1)
            stack_preds = None

        else:
            mean_preds = np.mean(X_cv, axis=1)
            mean_preds = [0 if x < .5 else 1 for x in mean_preds]
            stack_preds = None

        if self.stack:
            self.generalizer.fit(X_train, y)
            stack_preds = self.generalizer.predict(X_cv)

            if not self.isContinuous:
                stack_preds = np.add(np.sum(np.multiply(
                    self.generalizer.coef_, X_cv), axis=1), self.generalizer.intercept_)
                stack_cutoff = np.add(np.sum(self.generalizer.coef_), self.generalizer.intercept_) / 2.0
                stack_preds = [0 if x < stack_cutoff else 1 for x in np.asarray(stack_preds.reshape(-1))]

        return mean_preds, stack_preds

    def _get_model_cv_preds(self, model, X_train, y_train):
        """
        helper function for CV loop

        return CV model predictions
        """
        kfold = cross_validation.KFold(len(y_train), n_folds=5, random_state=42)
        stack_preds = []
        indexes_cv = []

        for stage0, stack in kfold:
            model.fit(np.asarray(X_train[stage0]), np.asarray(y_train[stage0]).reshape(-1))
            try:
                stack_preds.extend(list(model.predict_proba(
                    X_train[stack])[:, 1]))
            except AttributeError:
                stack_preds.extend(list(model.predict(X_train[stack])))
            indexes_cv.extend(list(stack))
        stack_preds = np.array(stack_preds)[sp.argsort(indexes_cv)]

        return stack_preds

    def _get_model_preds(self, model, feature_set, X_train, X_predict, y_train):
        """
        helper function for generalization

        return un-weighted model predictions for a given model,
        pickle model fit for speedup in prediction
        """
        model.fit(np.asarray(X_train[:, :]), np.asarray(y_train))

        if self.pickle_root_dir is not None:
            with open(
                    os.path.join(
                        self.pickle_root_dir, 'models-fits-' + stringify(model, feature_set) + '.pkl'), 'wb') as f:

                    pickle.dump(model, f)

        if self.isContinuous:
            try:
                model_preds = model.predict_proba(X_predict)[:, 1]
            except AttributeError:
                model_preds = model.predict(X_predict)

        return model_preds

    def fit_predict(self, y, data, features, train=None, predict=None,
                    show_steps=True):
        """
        Fit each model on the appropriate dataset, then return the average
        of their individual predictions. If train is specified, use a subset
        of the training set to train the models, then predict the outcome of
        either the remaining samples or (if given) those specified in cv.
        If train is omitted, train the models on the full training set, then
        predict the outcome of the full test set.

        Options:
        ------------------------------
        - y: numpy array. The full vector of the ground truths.
        - train: list. The indices of the elements to be used for training.
            If None, take the entire training set.
        - predict: list. The indices of the elements to be predicted.
        - show_steps: boolean. Whether to compute metrics after each stage
            of the computation.
        """
        y_train = y[train] if train is not None else y
        if train is not None and predict is None:
            predict = [i for i in range(len(y)) if i not in train]

        stage0_train = []
        stage0_predict = []

        for model, feature_set in self.models:
            X_train, X_predict = get_dataset(data, features, feature_set, train=train, cv=predict)

            model_preds = self._get_model_preds(
                model, feature_set, X_train, X_predict, y_train)
            stage0_predict.append(model_preds)

            # if stacking, compute cross-validated predictions on the train set
            if self.stack:
                model_cv_preds = self._get_model_cv_preds(
                    model, X_train, y_train)
                stage0_train.append(model_cv_preds)

            # verbose mode: compute metrics after every model computation
            if show_steps:
                if train is not None:

                    mean_preds, stack_preds = self._combine_predictions(
                        np.array(stage0_train).T, np.array(stage0_predict).T, y_train)

                    model_score = self.score_func(y[predict], stage0_predict[-1])
                    mean_score = self.score_func(y[predict], mean_preds)
                    stack_score = self.score_func(y[predict], stack_preds) \
                        if self.stack else 0

                    print "Model:", stringify(model, feature_set), \
                          "Model Score:", model_score,\
                          "Mean Score:", mean_score,\
                          "Stack Score:", stack_score

                else:
                    print("> used model %s:\n%s" % (stringify(
                        model, feature_set), model.get_params()))

        mean_preds, stack_preds = self._combine_predictions(
            np.array(stage0_train).T, np.array(stage0_predict).T,
            y_train)

        if self.for_model_select and self.stack:
            selected_preds = np.array(stage0_predict).T

        else:
            if self.stack:
                selected_preds = stack_preds

            else:
                selected_preds = mean_preds

        return selected_preds


class MLR(object):
    def __init__(self):
        self.coef_ = 0

    def fit(self, X, y):
        self.coef_ = sp.optimize.nnls(X, y)[0]
        self.coef_ = np.array(map(lambda x: x / sum(self.coef_), self.coef_))

    def predict(self, X):
        predictions = np.array(map(sum, self.coef_ * X))
        return predictions


def stringify(model, feature_set):
    """Given a model and a feature set, return a short string that will serve
    as identifier for this combination.
    Ex: (LogisticRegression(), "basic_s") -> "LR:basic_s"
    """
    return "%s-%s" % (sub("[a-z]", '', model.__class__.__name__), feature_set)


def get_dataset(data, features, feature_set='basic', train=None, cv=None):
    """
    Return the design matrices constructed with the specified feature set.
    If train is specified, split the training set according to train and
    the subsample's complement.
    """
    try:
        X_test = data[np.ix_(cv, features[feature_set])]
        X = data[np.ix_(train, features[feature_set])]
    except ValueError:
        X_test = data[:, features[feature_set]]
        X = data[:, features[feature_set]]

    return X, X_test


def find_params(model, feature_set, features,  y, data, subsample=None,
                grid_search=True):
    """
    Return parameter set for the model, either predefined
    or found through grid search.
    """
    model_name = model.__class__.__name__
    params = INITIAL_PARAMS.get(model_name, {})
    y = y if subsample is None else y[subsample]

    if grid_search and model_name in PARAM_GRID:
        print "Fitting params for :", model_name
        X, _ = get_dataset(data, features, feature_set, subsample)
        clf = GridSearchCV(model, PARAM_GRID[model_name], cv=5, n_jobs=4,
                           scoring="roc_auc")
        clf.fit(X, y)
        logger.info("found params (%s > %.4f): %s",
                    stringify(model, feature_set),
                    clf.best_score_, clf.best_params_)
        print "found params (%s > %.4f): %s" % (stringify(model, feature_set),
                                                clf.best_score_, clf.best_params_)
        params.update(clf.best_params_)

    return params
