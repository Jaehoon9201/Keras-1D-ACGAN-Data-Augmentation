
# ref1 : https://keras.io/getting_started/intro_to_keras_for_researchers/
# ref2 : https://www.linkedin.com/pulse/multi-task-supervised-unsupervised-learning-code-ibrahim-sobh-phd/
# ref3 : https://www.linkedin.com/pulse/supervised-variational-autoencoder-code-included-ibrahim-sobh-phd


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
from sklearn.preprocessing import StandardScaler
from keras.layers import Input, Dense, Lambda, Flatten, Reshape
from sklearn.metrics import roc_auc_score, roc_curve
from keras import metrics
import tensorflow.keras.backend as K


train_set = pd.read_csv('data/train_data.csv')
train_set = train_set.loc[:,['data1','data2','data3','data4', 'data5', 'data6', 'data7', 'data8', 'data9', 'data10', 'data11', 'data12', 'data13', 'data14', 'data15', 'data16',
                                                                   'data17', 'data18', 'data19', 'data20', 'data21', 'data22', 'data23', 'data24', 'data25', 'data26', 'data27', 'data28', 'sort']]

test_set = pd.read_csv('data/test_data.csv')
train_set = train_set.loc[:,['data1','data2','data3','data4', 'data5', 'data6', 'data7', 'data8', 'data9', 'data10', 'data11', 'data12', 'data13', 'data14', 'data15', 'data16',
                                                                   'data17', 'data18', 'data19', 'data20', 'data21', 'data22', 'data23', 'data24', 'data25', 'data26', 'data27', 'data28', 'sort']]
features = [ 'data11', 'data12', 'data13', 'data14', 'data15', 'data16']


err_train = train_set.loc[:, features].values
# err_train = StandardScaler().fit_transform(err_train)
status_train = train_set.loc[:,['sort']]
status_train = status_train.values.astype(np.int64)
status_train = np.ravel(status_train)
status_train_cat = to_categorical(status_train)


err_test = test_set.loc[:, features].values
# err_test = StandardScaler().fit_transform(err_test)
status_test = test_set.loc[:,['sort']]
status_test = status_test.values.astype(np.int64)
status_test = np.ravel(status_test)
status_test_cat = to_categorical(status_test)

status_test[(status_test > 0) & (status_test <= 6)] = 1
status_train[(status_train > 0) & (status_train <= 6)] = 1



import random
def shuffle_rows(arr, rows):
    np.random.shuffle(arr[rows[0]:rows[1] + 1])
err_train[(status_train == 1)] = shuffle(err_train[(status_train == 1)])
rows_to_delete = np.arange(0, len(err_train[(status_train == 1)])-len(err_train[(status_train == 0)]), step = 1)
err_train_1_reshaped = np.delete(err_train[(status_train == 1)], rows_to_delete, axis = 0)
status_train_1_reshaped = np.delete(status_train[(status_train == 1)], rows_to_delete, axis = 0)
err_train_reshaped = np.append(err_train[status_train != 1], err_train_1_reshaped , axis=0)
status_train_reshaped = np.append(status_train[(status_train != 1)], status_train_1_reshaped , axis=0)
err_train = err_train_reshaped
status_train = status_train_reshaped

err_test[(status_test == 1)] = shuffle(err_test[(status_test == 1)])
rows_to_delete = np.arange(0, len(err_test[(status_test == 1)])-len(err_test[(status_test == 0)]), step = 1)
err_test_1_reshaped = np.delete(err_test[(status_test == 1)], rows_to_delete, axis = 0)
status_test_1_reshaped = np.delete(status_test[(status_test == 1)], rows_to_delete, axis = 0)
err_test_reshaped = np.append(err_test[status_test != 1], err_test_1_reshaped , axis=0)
status_test_reshaped = np.append(status_test[(status_test != 1)], status_test_1_reshaped , axis=0)
err_test = err_test_reshaped
status_test = status_test_reshaped
#■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■

x_train = err_train
y_train = status_train

x_test = err_test
y_test = status_test


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)


anom_mask1 = (y_train == 1)
anomaly_test1 = x_train[anom_mask1]
x_train = x_train[~anom_mask1]
y_train = y_train[~anom_mask1]

#anom_mask2 = (y_test == (1 and 5 and 6 and 7 and 8 and 9))
anom_mask2 = (y_test == 1)
anomaly_test2 = x_test[anom_mask2]
x_test = x_test[~anom_mask2]
y_test = y_test[~anom_mask2]

anomaly_test = np.concatenate((anomaly_test1, anomaly_test2), axis = 0)
anomaly_test = shuffle(anomaly_test)
rows_to_delete = np.arange(0, len(anomaly_test)-len(x_test), step = 1)
anomaly_test = np.delete(anomaly_test, rows_to_delete, axis = 0)
print('Training sets', x_train.shape, 'Testing sets',x_test.shape, 'Anomaly sets', anomaly_test.shape)

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

latent_dim = 2
original_dim = 6
dense1 = 8
dense2 = 4

# ■■■■■■■ Load model and eval ■■■■■■■■
var_str = 'epoch batch'
var1 = 200
var2 = 20
train_epoch = var1
batch_size = var2

learn_rate = 4e-5
original_shape = x_train.shape[1:]



# ■■■■■■■■■■■■■■■■■■
# ■■■■■■   VAE  ■■■■■■■
# ■■■■■■■■■■■■■■■■■■
in_layer = Input(shape=original_shape)
x = Flatten()(in_layer)
h = Dense(dense1, activation='tanh')(x)
h = Dense(dense2, activation = 'tanh')(h)

z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)
epsilon_std = 1
def sampling(args):
    z_mean, z_log_var = args
    epsilon = tf.random.normal(shape=(tf.shape(z_mean)[0], latent_dim), mean=0.,
                              stddev=epsilon_std)
    return z_mean + tf.exp(z_log_var / 2) * epsilon

# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# we instantiate these layers separately so as to reuse them later
decoder_f = Dense(dense2, activation='tanh')
decoder_h = Dense(dense1, activation='tanh')
decoder_mean = Dense(original_dim, activation='tanh')

f_decoded = decoder_f(z)
h_decoded = decoder_h(f_decoded)
x_decoded_mean = decoder_mean(h_decoded)
x_decoded_img = Reshape(original_shape)(x_decoded_mean)

# instantiate VAE model
vae = Model(in_layer, x_decoded_img)

# Compute VAE loss
xent_loss = original_dim * metrics.binary_crossentropy(x, x_decoded_mean)
kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
vae_loss = K.mean(0.5* xent_loss + 0.5*kl_loss)

vae.add_loss(vae_loss)
vae.compile(optimizer='adam')
vae.summary()

vae.fit(x_train,
        shuffle=True,
        epochs=train_epoch,
        batch_size=batch_size,
        validation_data=(anomaly_test, None))

vae.save('%s = %d %d vae_dense2_model.hdf5' % (var_str, var1, var2))

encoder = Model(in_layer, z_mean)

print(x_test)
print(encoder(x_test))

x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
anomaly_encoded = encoder.predict(anomaly_test, batch_size=batch_size)

fig, m_axs = plt.subplots(latent_dim,latent_dim, figsize=(latent_dim*5, latent_dim*5))
if latent_dim == 1:
    m_axs = [[m_axs]]

for i, n_axs in enumerate(m_axs, 0):
    for j, c_ax in enumerate(n_axs, 0):
        c_ax.scatter(np.concatenate([x_test_encoded[:, i], anomaly_encoded[:,i]],0),
                     np.concatenate([x_test_encoded[:, j], anomaly_encoded[:,j]],0),
                     c=(['g']*x_test_encoded.shape[0])+['r']*anomaly_encoded.shape[0], alpha = 0.1)

model_mse = lambda x: np.sqrt(np.mean(np.square(x-vae.predict( x, batch_size = batch_size)), (1)))

from sklearn.metrics import roc_auc_score, roc_curve
mse_score = np.concatenate([model_mse(x_test), model_mse(anomaly_test)], 0)
print('mse-score 1')
print(mse_score)
print(len(mse_score))
true_label = [0]*x_test.shape[0]+[1]*anomaly_test.shape[0]
print(true_label)

if roc_auc_score(true_label, mse_score)<0.5:
    mse_score *= -1
print('mse-score 2')
print(mse_score)


# mse_score 작으면 : 0 (정상)  / 크면 : 1(비정상)
fpr, tpr, thresholds = roc_curve(true_label, mse_score)


print(abs(thresholds))
auc_score = roc_auc_score(true_label, mse_score)
fig, ax1 = plt.subplots(1, 1, figsize = (6, 6))
ax1.plot(fpr, tpr, 'b.-', label = 'ROC Curve (%4.4f)' %  auc_score)
ax1.plot(fpr, fpr, 'k-', label = 'Random Guessing')
ax1.legend();
plt.show()


# # ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
# # MSE 사사용 counting
# # ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■

train_mse = model_mse(x_train)
test_mse = model_mse(x_test)
# mse_threshold = np.max(train_mse)
# mse_threshold = 0.2
mse_threshold = 0.7
idx = np.where(test_mse > mse_threshold)
print('Normal state but classified as anomaly')
print(np.count_nonzero(idx))

anomaly_mse = model_mse(anomaly_test)
idx = np.where(anomaly_mse > mse_threshold)
print('Anomaly state and classified as anomaly')
print(np.count_nonzero(idx))


fpr = fpr.reshape(-1, 1)
tpr = tpr.reshape(-1, 1)
thresholds = thresholds.reshape(-1, 1)
print(tpr.shape)
fpr_tpr_thre = np.append(fpr, tpr, axis = 1)
fpr_tpr_thre = np.append(fpr_tpr_thre, thresholds, axis = 1)
# np.savetxt('fpr_tpr_thre.csv',fpr_tpr_thre,delimiter=",")