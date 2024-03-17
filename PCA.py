from tables import Cols
import pandas as pd

cols=['petal_length','petal_width','sepal_length','sepal_width','species_type']
data=pd.read_csv('iris.data',names=cols,header=None)

X=data.iloc[:,0:4]
y = data.iloc[:,-1]
print(y.shape)
print(X.shape)

#splitting data
from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(X, y,
                                                    train_size=0.7,
                                                    test_size=0.3,
                                                    random_state=2)
#train_X.head(), test_X.head(), train_y.head(), test_y.head()

import numpy as np
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(train_X)
train_X=pca.transform(train_X)
test_X=pca.transform(test_X)
print(pca.explained_variance_ratio_)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
classifier = KNeighborsClassifier(n_neighbors=1)
classifier.fit(train_X, train_y)
pred_y = classifier.predict(test_X)
# print(pred_y)
# print('accuracy score after aplying KNN by library is  ' ,classifier.score(test_X, test_y))
print('accuracy score after aplying KNN by library is  ',accuracy_score(test_y,pred_y))