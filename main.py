import os
import numpy as np
import pandas as pd
from sklearn import linear_model, ensemble
from sklearn import metrics
from sklearn import cross_validation
from sklearn.decomposition import TruncatedSVD
from stumbleupon.src.su_code import ModelEnsemble
from sklearn.feature_extraction.text import TfidfVectorizer
from stumbleupon.src.su_code.util import LemmaTokenizer, create_test_submission


def train_su_model(path_to_data=None):
    """
    transforming link tf-idf text content to a dense matrix using LSA, fit an ensemble of different
    machine learning models using stacked generalization, which combines the CV predictions of each model
    as variable inputs to a second classifier, whose coefficients effectively weigh the importance of each model.
    """
    train_text = list(np.array(pd.read_csv(path_to_data + "/train_text.csv"))[:, 1])
    test_text = list(np.array(pd.read_csv(path_to_data + "/test_text.csv"))[:, 1])

    y = np.transpose(list(np.array(pd.read_csv(path_to_data + "/train_text.csv"))[:, -1]))

    X_all = train_text + test_text
    X_all = [X_all[x].decode('ascii', 'ignore').encode('utf-8') for x in range(len(X_all))]

    lentrain = len(train_text)

    cntVec = TfidfVectorizer(min_df=3,  max_features=None, strip_accents='unicode',
                             analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 2),
                             sublinear_tf=True, use_idf=True, tokenizer=LemmaTokenizer())

    print "transforming data"
    X_word = cntVec.fit_transform(X_all)

    #Partial SVD decomp, (LSA)

    trans_LSA = TruncatedSVD(n_components=150, algorithm='randomized', n_iter=5)

    X_LSA = trans_LSA.fit_transform(X_word)
    X_LSA_train = X_LSA[:lentrain]

    features = {'LSA': (range(0, X_LSA.shape[1]))}


    SEED = 42

    selected_models = [
        "LR:LSA",
        "GBC:LSA"
    ]

    models = []
    for item in selected_models:
        model_id, dataset = item.split(':')
        model = {'LR': linear_model.LogisticRegression,
                 'GBC': ensemble.GradientBoostingClassifier,
                 'RFC': ensemble.RandomForestClassifier,
                 'ADA': ensemble.AdaBoostClassifier,
                 'ETC': ensemble.ExtraTreesClassifier}[model_id]()
        model.set_params(random_state=SEED)
        models.append((model, dataset))

    print "finding optimal meta-parameters for each model in ensemble"
    for model, feature_set in models:
        model.set_params(
            **ModelEnsemble.find_params(model, feature_set, features, y, X_LSA_train, subsample=None, grid_search=True))

    print "fitting ensemble"
    clf = ModelEnsemble.EnsembleGeneralization(models, metrics.roc_auc_score, for_model_select=True, stack=True)

    k_fold = cross_validation.KFold(X_LSA_train.shape[0], n_folds=5, shuffle=True, random_state=SEED)
    mean_auc = 0

    for train, cv in k_fold:
        stack_cv_preds = clf.fit_predict(y=y, data=X_LSA_train, features=features, train=train, predict=cv, show_steps=True)
        ensemble_cv_fit = ModelEnsemble.MLR()
        ensemble_cv_fit.fit(stack_cv_preds, y[cv])
        fpr, tpr, _ = metrics.roc_curve(y[cv], ensemble_cv_fit.predict(stack_cv_preds))
        roc_auc = metrics.auc(fpr, tpr)
        mean_auc += roc_auc

    print("CV AUC of ensemble: ", mean_auc / 5.0)

    clf = ModelEnsemble.EnsembleGeneralization(models, metrics.roc_auc_score, for_model_select=False, stack=True)

    final_predictions = clf.fit_predict(
        y=y, data=X_LSA, train=range(len(y)), predict=range(len(y), X_LSA.shape[0]), features=features, show_steps=False)
    
    urlid = np.array(pd.read_csv(path_to_data + "/test_text.csv"))[:, 0]
    create_test_submission(urlid, '/home/dylanjf/PycharmProjects/kaggle/stumbleupon/submission_final.csv',
                           final_predictions)
    print "Submission complete!"
    
if __name__ == "__main__":
    path_to_data = os.getcwd()
    try:
        train_su_model(path_to_data)
    except:
        print "Failed to train model, please make sure the data is stored in ", path_to_data
