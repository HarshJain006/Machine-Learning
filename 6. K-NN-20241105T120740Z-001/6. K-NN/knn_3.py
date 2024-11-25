import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
from sklearn.compose import make_column_transformer 
from sklearn.compose import make_column_selector
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.pipeline import Pipeline
import os 
os.chdir(r"C:\Training\Academy\Statistics (Python)\Datasets")

boston = pd.read_csv("Boston.csv")
X = boston.drop('medv', axis=1)
y = boston['medv']

ohe = OneHotEncoder(sparse_output=False, drop='first').set_output(transform='pandas')
scaler_mm = MinMaxScaler()
scaler_std = StandardScaler()
ct = make_column_transformer(('passthrough', make_column_selector(dtype_exclude=object)  ),
                             (ohe, make_column_selector(dtype_include=object) ),
                            verbose_feature_names_out=False).set_output(transform='pandas')


####### Grid Search 
knn = KNeighborsRegressor()
pipe = Pipeline([('CT',ct),('SCL',None),('KNN',knn)])
kfold = KFold(n_splits=5, random_state=24, 
                        shuffle=True)
params = {'KNN__n_neighbors': np.arange(1,16),
          'KNN__metric':['cityblock','minkowski','manhattan','haversine'],
          'SCL':[scaler_mm, scaler_std, None]}

gcv = GridSearchCV(pipe, param_grid=params, 
       scoring='r2', cv=kfold, verbose=3)
gcv.fit(X,y)
print(gcv.best_params_)
print(gcv.best_score_)

pd_cv = pd.DataFrame(gcv.cv_results_)
print(pd_cv.shape)

