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
import matplotlib.colors as mcolors


df = pd.read_csv(r'data\data.csv')
df = df.loc[:,['data1','data2','data3','data4','data5','data6', 'data7', 'data8', 'data9', 'data10', 'state']]
features = ['data1','data2','data3','data4','data5','data6', 'data7', 'data8', 'data9', 'data10']
x = df.loc[:, features].values
y = df.loc[:,['state']].values
x = StandardScaler().fit_transform(x)


from sklearn.decomposition import PCA
pca = PCA(n_components=2) # 주성분을 몇개로 할지 결정
printcipalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data=printcipalComponents, columns = ['principal component1', 'principal component2'])

finalDf = pd.concat([principalDf, df[['state']]], axis = 1)

import matplotlib.pyplot as plt
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('First Principal Component', fontsize = 15)
ax.set_ylabel('Second Principal Component', fontsize = 15)

states = [0,1,2,3,4,5,6]
colors = [mcolors.CSS4_COLORS['forestgreen'],mcolors.CSS4_COLORS['red'],
         mcolors.CSS4_COLORS['tomato'],mcolors.CSS4_COLORS['blueviolet'],
         mcolors.CSS4_COLORS['darkorange'],mcolors.CSS4_COLORS['goldenrod'],
         mcolors.CSS4_COLORS['hotpink']]

for state, color in zip(states, colors):
    indicesToKeep = finalDf['state'] == state
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component1']
               , finalDf.loc[indicesToKeep, 'principal component2']
               , c = color
               , s = 15)
labl = [ 'Normal',  'A-phase fault',  'B-phase fault',  'C-phase fault',
         'AB-phase fault', 'BC-phase fault',  'CA-phase fault']
ax.legend(labl)
ax.grid()
plt.show()