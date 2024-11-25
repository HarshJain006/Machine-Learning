import pandas as pd
from sklearn.naive_bayes import BernoulliNB
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
import os 
os.chdir(r"C:\Training\Academy\Statistics (Python)\Cases\Cancer")

df = pd.read_csv("Cancer.csv", index_col=0)
X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, 
        random_state=24, test_size=0.3, stratify=y)

ohe = OneHotEncoder(sparse_output=False, 
      handle_unknown='ignore').set_output(transform='pandas')
nb = BernoulliNB()
pipe = Pipeline([('OHE', ohe),('NB',nb)])
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
print(classification_report(y_test, y_pred))

y_pred_prob = pipe.predict_proba(X_test)[:,1]
print(roc_auc_score(y_test, y_pred_prob))

##############################################
kfold = StratifiedKFold(n_splits=5, shuffle=True, 
                        random_state=24)
params = {'NB__alpha':np.linspace(0.001, 3, 10)}
gcv = GridSearchCV(pipe, param_grid=params,
       scoring='roc_auc', cv=kfold)
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)
