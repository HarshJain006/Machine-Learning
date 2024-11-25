import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.compose import make_column_transformer 
from sklearn.compose import make_column_selector
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
import os 
os.chdir(r"C:\Training\Kaggle\Competitions\Playground Competitions\Smoker Biosignals")

df = pd.read_csv("train.csv", index_col=0)
X = df.drop('smoking', axis=1)
y = df['smoking']

lr = LogisticRegression(random_state=24)

kfold = StratifiedKFold(n_splits=5, random_state=24, 
                        shuffle=True)
params = {'solver':['lbfgs','liblinear',
          'newton-cg','newton-cholesky',
          'sag','saga']}
gcv = GridSearchCV(lr, param_grid=params, 
      scoring='roc_auc',cv=kfold, verbose=3)
gcv.fit(X,y)
print(gcv.best_params_)
print(gcv.best_score_)

pd_cv = pd.DataFrame(gcv.cv_results_)
print(pd_cv.shape)

##### Inferencing with logistic
best_model = gcv.best_estimator_
test = pd.read_csv("test.csv", index_col=0)

y_pred_prob = best_model.predict_proba(test)[:,1]

submit = pd.read_csv("sample_submission.csv")
submit['smoking'] = y_pred_prob

submit.to_csv("sbt_5_Nov_lr.csv", index=False)
