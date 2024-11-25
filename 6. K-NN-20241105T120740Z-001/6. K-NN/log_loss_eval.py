import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.compose import make_column_transformer 
from sklearn.compose import make_column_selector
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
import os 
os.chdir(r"C:\Training\Academy\Statistics (Python)\Cases\human-resources-analytics")

hr = pd.read_csv("HR_comma_sep.csv")
X = hr.drop('left', axis=1)
y = hr['left']

X_train, X_test, y_train, y_test = train_test_split(X, y, 
        random_state=24, test_size=0.3, stratify=y)

ohe = OneHotEncoder(sparse_output=False, drop='first').set_output(transform='pandas')
scaler_mm = MinMaxScaler()
scaler_std = StandardScaler()
ct = make_column_transformer(('passthrough', make_column_selector(dtype_exclude=object)  ),
                             (ohe, make_column_selector(dtype_include=object) ),
                            verbose_feature_names_out=False).set_output(transform='pandas')
lr = LogisticRegression(random_state=24)
pipe = Pipeline([('CT',ct),('SCL',None),('LR',lr)])

pipe.fit(X_train, y_train)
y_pred_prob = pipe.predict_proba(X_test)
print(log_loss(y_test, y_pred_prob))

#### K-FOLDS

kfold = StratifiedKFold(n_splits=5, random_state=24, 
                        shuffle=True)
params = {'LR__solver':['lbfgs','liblinear',
          'newton-cg','newton-cholesky',
          'sag','saga'],
          'LR__C':np.linspace(0.001, 10, 20),
          'SCL':[scaler_mm, scaler_std, None]}
# gcv = GridSearchCV(pipe, param_grid=params,
#                    scoring='roc_auc',
#                    cv=kfold, verbose=3)
gcv = GridSearchCV(pipe, param_grid=params,
                    scoring='neg_log_loss',
                    cv=kfold, verbose=3)
gcv.fit(X,y)
print(gcv.best_params_)
print(gcv.best_score_)

pd_cv = pd.DataFrame(gcv.cv_results_)
print(pd_cv.shape)
