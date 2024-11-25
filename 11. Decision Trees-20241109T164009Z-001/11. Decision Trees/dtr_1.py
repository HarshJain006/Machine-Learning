import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

x = np.array([[2,4],
              [3,5],
              [12,18],
              [15,20],
              [34,56],
              [35,60],
              [78, 26],
              [80, 23]])
y = np.array([45,48,100,109,56,50,115,108])

p_df = pd.DataFrame(x, columns=['x1', 'x2'])
p_df['y'] = y

sns.scatterplot(data=p_df, x='x1',y='x2',
                hue='y')
plt.show()

X = p_df[['x1','x2']]
y = p_df['y']

dtr = DecisionTreeRegressor(random_state=24)
dtr.fit(X, y)

plt.figure(figsize=(25,10))
plot_tree(dtr,feature_names=list(X.columns),
               filled=True,fontsize=18)
plt.show() 

x_test = pd.DataFrame({'x1':[50,20,30],
                       'x2':[5, 50,30]})
print(dtr.predict(x_test))

## data 

x = np.array([[2,4],
              [3,5],
              [12,18],
              [15,20],
              [34,56],
              [35,60],
              [78, 26],
              [80, 23],
              [40, 20],
              [50, 30],
              [40, 30],
              [10, 30],
              [80,40],
              [5,40]])
y = np.array([45,48,100,109,56,50,115,108, 46,43,36,106, 109,130])

p_df = pd.DataFrame(x, columns=['x1', 'x2'])
p_df['y'] = y

sns.scatterplot(data=p_df, x='x1',y='x2',
                hue='y')
plt.show()

X = p_df[['x1','x2']]
y = p_df['y']


dtr = DecisionTreeRegressor(random_state=24)
dtr.fit(X, y)

plt.figure(figsize=(25,10))
plot_tree(dtr,feature_names=list(X.columns),
               filled=True,fontsize=18)
plt.show()


X_train, X_test, y_train, y_test = train_test_split(X, y,
                                   random_state=24)

dtr = DecisionTreeRegressor(random_state=24, max_depth=2)
dtr.fit(X_train, y_train)

plt.figure(figsize=(35,10))
plot_tree(dtr,feature_names=list(X.columns),
               filled=True,fontsize=18)
plt.show()
y_pred = dtr.predict(X_test)
print(r2_score(y_test, y_pred))


# tst = np.array([[30,40],
#                 [5, 20],
#                 [50, 50]])

# X_test = pd.DataFrame(tst,columns=['x1', 'x2'] )
# dtc.predict(X_test)
