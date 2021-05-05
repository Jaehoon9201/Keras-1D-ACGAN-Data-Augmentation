# https://machinelearningmastery.com/principal-components-analysis-for-dimensionality-reduction-in-python/
# https://ai-hyu.com/python-pca-%EA%B5%AC%ED%98%84%ED%95%98%EA%B8%B0/
# https://scipython.com/blog/plotting-the-decision-boundary-of-a-logistic-regression-model/
#evaluate pca with logistic regression algorithm for classification
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
import sklearn.linear_model

from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import load_breast_cancer
import matplotlib.colors as mcolors
from sklearn.model_selection import train_test_split

df = pd.read_csv(r'data\data.csv')
df = df.loc[:,['data1','data2','data3','data4','data5','data6', 'data7', 'data8', 'data9', 'data10', 'state']]
features = ['data1','data2','data3','data4','data5','data6', 'data7', 'data8', 'data9', 'data10']

X = df.loc[:, features].values
y = df.loc[:,['state']]
y = y.to_numpy()
y = y.reshape(-1)
print(y.shape)


# ■■■■■■■■■■ Method 1 for scaler ■■■■■■■■■■
# Standardizing the features
# X = StandardScaler().fit_transform(X)
# #X = MinMaxScaler().fit_transform(X)

# ■■■■■■■■■■ Method 2 for scaler ■■■■■■■■■■
# calculate the mean of each column : StandardScaler
X = (X - np.mean(X.T, axis = 1)) / np.std(X.T,axis = 1)

# ■■■■■■■■■■ Method 1 for PCA + classification ■■■■■■■■■■
# # define the pipeline
# steps = [('pca', PCA(n_components=2)), ('m', LogisticRegression())]
# model = Pipeline(steps=steps)
# # evaluate model
# cv = RepeatedStratifiedKFold(n_splits=2, n_repeats=3, random_state=1)
# n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
# # report performance
# print('Accuracy: %.4f (%.4f)' % (mean(n_scores), std(n_scores)))

# ■■■■■■■■■■ Method 2 for PCA + classification ■■■■■■■■■■
def PCA_MANUAL():
    cov_X = np.cov(X.T)
    eigen_val, eigen_vec = np.linalg.eig(cov_X)
    z1 = (eigen_vec[:,0][0] * X[:,0] + eigen_vec[:,0][1] * X[:,1] + eigen_vec[:,0][2] * X[:,2]
        + eigen_vec[:,0][3] * X[:,3] + eigen_vec[:,0][4] * X[:,4] + eigen_vec[:,0][5] * X[:,5]
        + eigen_vec[:,0][6] * X[:,6] + eigen_vec[:,0][7] * X[:,7] + eigen_vec[:,0][8] * X[:,8]
        + eigen_vec[:,0][9] * X[:,9] )
    z2 = (eigen_vec[:,1][0] * X[:,0] + eigen_vec[:,1][1] * X[:,1] + eigen_vec[:,1][2] * X[:,2]
        + eigen_vec[:,1][3] * X[:,3] + eigen_vec[:,1][4] * X[:,4] + eigen_vec[:,1][5] * X[:,5]
        + eigen_vec[:,1][6] * X[:,6] + eigen_vec[:,1][7] * X[:,7] + eigen_vec[:,1][8] * X[:,8]
        + eigen_vec[:,1][9] * X[:,9] )
    z3 = (eigen_vec[:, 2][0] * X[:, 0] + eigen_vec[:, 2][1] * X[:, 1] + eigen_vec[:, 2][2] * X[:, 2]
        + eigen_vec[:, 2][3] * X[:, 3] + eigen_vec[:, 2][4] * X[:, 4] + eigen_vec[:, 2][5] * X[:, 5]
        + eigen_vec[:, 2][6] * X[:, 6] + eigen_vec[:, 2][7] * X[:, 7] + eigen_vec[:, 2][8] * X[:, 8]
        + eigen_vec[:, 2][9] * X[:, 9])
    # z1 = (eigen_vec[:, 0][0] * X[:, 0] + eigen_vec[:, 0][1] * X[:, 1] + eigen_vec[:, 0][2] * X[:, 2]
    #       + eigen_vec[:, 0][3] * X[:, 3] + eigen_vec[:, 0][4] * X[:, 4] + eigen_vec[:, 0][5] * X[:, 5]
    #       )
    # z2 = (eigen_vec[:, 1][0] * X[:, 0] + eigen_vec[:, 1][1] * X[:, 1] + eigen_vec[:, 1][2] * X[:, 2]
    #       + eigen_vec[:, 1][3] * X[:, 3] + eigen_vec[:, 1][4] * X[:, 4] + eigen_vec[:, 1][5] * X[:, 5]
    #       )
    # z3 = (eigen_vec[:, 2][0] * X[:, 0] + eigen_vec[:, 2][1] * X[:, 1] + eigen_vec[:, 2][2] * X[:, 2]
    #       + eigen_vec[:, 2][3] * X[:, 3] + eigen_vec[:, 2][4] * X[:, 4] + eigen_vec[:, 2][5] * X[:, 5]
    #       )
    pca_res = np.vstack([z1,z2,z3]).T
    return pca_res

X_pca = PCA_MANUAL()

ex_variance=np.var(X_pca,axis=0)
ex_variance_ratio = ex_variance/np.sum(ex_variance)
ex_variance_ratio


Xax = X_pca[:,0]
Yax = X_pca[:,1]
Zax = X_pca[:,2]

cdict = {0:mcolors.CSS4_COLORS['forestgreen'],1:mcolors.CSS4_COLORS['red'],
         2:mcolors.CSS4_COLORS['tomato'],3:mcolors.CSS4_COLORS['blueviolet'],
         4:mcolors.CSS4_COLORS['darkorange'],5:mcolors.CSS4_COLORS['goldenrod'],
         6:mcolors.CSS4_COLORS['hotpink']}
labl = {0:'Normal',1:'A-phase fault',2:'B-phase fault',3:'C-phase fault',
                4:'AB-phase fault',5:'BC-phase fault',6:'CA-phase fault'}
marker = {0:'o',1:'o',2:'o',3:'o',4:'o',5:'o',6:'o'}
alpha = {0:.3,1:.3,2:.3,3:.3,4:.3,5:.3,6:.3}

fig = plt.figure(figsize=(14,10))
ax = fig.add_subplot(111, projection='3d')

fig.patch.set_facecolor('white')
for l in np.unique(y):
 ix=np.where(y==l)
 ax.scatter(Xax[ix], Yax[ix], Zax[ix], c=cdict[l], s=12,
           label=labl[l], marker=marker[l], alpha=alpha[l])
# for loop ends
ax.set_xlabel("First Principal Component", fontsize=10)
ax.set_ylabel("Second Principal Component", fontsize=10)
ax.set_zlabel("Third Principal Component", fontsize=10)
# ax.set_xlim([-4, 6])
# ax.set_ylim([-3, 3.5])
# ax.set_zlim([-3, 3])
ax.legend()
plt.show()
