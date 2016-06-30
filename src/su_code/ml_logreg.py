import numpy as np
import pandas as pd
import sklearn.linear_model as lm
from sklearn.cross_validation import KFold
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer

traindata = list(np.array(pd.read_csv("C:/Users/dylanjf/Desktop/kaggle/stumbleupon/train_text.csv"))[:,1])
testdata = list(np.array(pd.read_csv("C:/Users/dylanjf/Desktop/kaggle/stumbleupon/test_text.csv"))[:,1])
y = np.array(pd.read_csv("C:/Users/dylanjf/Desktop/kaggle/stumbleupon/train_text.csv"))[:,-1]

tfv = TfidfVectorizer(min_df=3,  max_features=None, strip_accents='unicode',  
                      analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 2),
                      use_idf=1, smooth_idf=1,sublinear_tf=1)

model = lm.LogisticRegression(penalty='l2', dual=True, tol=0.0001, 
                            C=1, fit_intercept=True, intercept_scaling=1.0, 
                            class_weight=None, random_state=None)

X_all = traindata + testdata
X_all = [X_all[x].decode('ascii', 'ignore').encode('utf-8') for x in range(len(X_all))]

lentrain = len(traindata)

print "fitting pipeline"
tfv.fit(X_all)
print "transforming data"
X_all = tfv.transform(X_all)

X = X_all[:lentrain]
X_test = X_all[lentrain:]

########3 rep 10 fold CV to determine feature sparsity percentage via RFE########

def log_reg_RFE(X, y, model, n_folds=10, n_reps=1, seed=42):
    """
    performs recursive feature elimination via log_reg coefficient removal.
    verifies via 10 fold CV.  allows for repeated 10CV.  seed for reproducability. Lightly verbose.
    returns the best percentage subset of the data columns, verified by AUC.
    """
    seed = seed
    mean_auc = [0] * 100

    for i in range(1, n_reps+1):

        cv_split = KFold(len(y), n_folds=n_folds, indices=True, random_state=seed*i)

        for train_index, test_index in cv_split:

            X_cv_train, X_cv_test = X[train_index], X[test_index]
            Y_cv_train, Y_cv_test = y[train_index], y[test_index]

            model_fit = model.fit(X_cv_train, Y_cv_train)
            coef = model.coef_.ravel(model_fit)
            important_coef = np.argsort(np.abs(coef))

            for j in range(len(mean_auc)):

                important_coef_subset = important_coef[-int(len(important_coef) * (1 - j/100.0)):]
                X_cv_train_subset, X_cv_test_subset = X_cv_train[:, important_coef_subset], X_cv_test[:, important_coef_subset]

                model.fit(X_cv_train_subset, Y_cv_train)
                pred = model.predict_proba(X_cv_test_subset)[:, 1]
                mean_auc[j] += metrics.roc_auc_score(Y_cv_test, pred) / float(n_folds)

        print "Fold set %d complete." % i

    mean_auc = [mean_auc[x] / n_reps for x in range(len(mean_auc))]
    best_pct = np.argsort(mean_auc)[-1]

    #fitting set with optimal amount of feature reduction

    important_coef_RFE = important_coef[-int(len(important_coef) * (1 - best_pct/100.0)):]

    return important_coef_RFE, mean_auc

best_col, mean_auc = log_reg_RFE(X=X, y=y, model=model, n_folds=10, n_reps=1, seed=42)

X_bestRFE = X[:,best_col]
X_test_bestRFE = X_test[:,best_col]

log_RFE_fit = model.fit(X_bestRFE, y)
pred_RFE = model.predict_proba(X_test_bestRFE)[:,1]


#writing csv for submission

urlid = np.array(pd.read_csv("C:/Users/dylanjf/Desktop/kaggle/stumbleupon/test_text.csv"))[:,0]

def create_test_submission(filename, prediction):
    content = ['urlid,label']
    for i, p in enumerate(prediction):
        content.append('%i,%f' %(urlid[i],p))
    f = open(filename, 'w')
    f.write('\n'.join(content))
    f.close()
    print 'Saved'

create_test_submission('C:/Users/dylanjf/Desktop/kaggle/stumbleupon/submission3.csv',pred_RFE)