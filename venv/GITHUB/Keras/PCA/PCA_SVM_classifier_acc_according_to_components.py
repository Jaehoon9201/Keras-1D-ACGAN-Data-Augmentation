from keras.datasets import mnist
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, confusion_matrix
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow.python.keras.losses import binary_crossentropy
import keras as K
import keras
import numpy as np
from tensorflow.python.keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns
from keras.layers import Input, Dense
from keras.models import Model
import pandas as pd
import numpy as np
import glob
import os
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import StandardScaler

from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from matplotlib import pyplot
from sklearn import svm

df = pd.read_csv(r'data\data.csv')
df = df.loc[:,['data1','data2','data3','data4','data5','data6', 'data7', 'data8', 'data9', 'data10', 'state']]
features = ['data1','data2','data3','data4','data5','data6', 'data7', 'data8', 'data9', 'data10']

X = df.loc[:, features].values
y = df.loc[:,['state']]
y = y.to_numpy()
y = y.reshape(-1)
print(y.shape)


# ■■■■■■■■■■ Method 1 for scaler ■■■■■■■■■■
# X = StandardScaler().fit_transform(X)
# #X = MinMaxScaler().fit_transform(X)

# ■■■■■■■■■■ Method 2 for scaler ■■■■■■■■■■
X = (X - np.mean(X.T, axis = 1)) / np.std(X.T,axis = 1)


# ■■■■■■■■■■ Method 1 for PCA + classification ■■■■■■■■■■
# define the pipeline
#steps = [('pca', PCA(n_components=2)), ('m', LogisticRegression())]
steps = [('pca', PCA(n_components=2)), ('svm', svm.SVC(kernel= 'rbf'))]

model = Pipeline(steps=steps)
# evaluate model
cv = RepeatedStratifiedKFold(n_splits=2, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
# report performance
print('Accuracy: %.4f (%.4f)' % (mean(n_scores), std(n_scores)))

# ■■■■■■■■■■ Method 2 for PCA + classification ■■■■■■■■■■
# def PCA_MANUAL():
#     cov_X = np.cov(X.T)
#     eigen_val, eigen_vec = np.linalg.eig(cov_X)
#     z1 = eigen_vec[:,0][0] * X[:,0] + eigen_vec[:,0][1] * X[:,1] + eigen_vec[:,0][2] * X[:,2]
#     z2 = eigen_vec[:,1][0] * X[:,0] + eigen_vec[:,1][1] * X[:,1] + eigen_vec[:,1][2] * X[:,2]
#     pca_res = np.vstack([z1,z2]).T
#     return pca_res
# steps = [('pca', PCA_MANUAL()), ('m', LogisticRegression())]
# model = Pipeline(steps=steps)
# # evaluate model
# cv = RepeatedStratifiedKFold(n_splits=2, n_repeats=3, random_state=1)
# n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
# # report performance
# print('Accuracy: %.4f (%.4f)' % (mean(n_scores), std(n_scores)))




def get_models():
	models = dict()
	for i in range(1,10+1):
		#steps = [('pca', PCA(n_components=i)), ('m', LogisticRegression())]
		steps = [('pca', PCA(n_components=i)), ('svm', svm.SVC(kernel='rbf'))]
		models[str(i)] = Pipeline(steps=steps)
	return models

# evaluate a given model using cross-validation
def evaluate_model(model, X, y):
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
	return scores


models = get_models()
# evaluate the models and store results
results, names = list(), list()
for name, model in models.items():
	scores = evaluate_model(model, X, y)
	results.append(scores)
	names.append(name)
	print('>%s %.4f (%.4f)' % (name, mean(scores), std(scores)))

pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.xticks(rotation=45)
plt.grid(True)
pyplot.show()