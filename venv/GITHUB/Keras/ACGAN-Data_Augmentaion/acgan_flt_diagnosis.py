# Refence : https://machinelearningmastery.com/how-to-develop-an-auxiliary-classifier-gan-ac-gan-from-scratch-with-keras/

from numpy import zeros
from numpy import ones
from numpy import expand_dims
from numpy.random import randn
from numpy.random import randint
from keras.datasets.fashion_mnist import load_data
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import Activation
from keras.layers import Concatenate
from keras.initializers import RandomNormal
from matplotlib import pyplot
import keras as K
import glob
import os


import tensorflow as tf
import pandas as pd
import numpy as np
import glob
import os
from keras.utils import to_categorical
from sklearn.metrics import classification_report


df2 = pd.read_csv('data/train_data.csv')
err_train = df2.values[:, :-1]
status_train = df2.values[:, -1].astype(np.int64)
status_train_cat = to_categorical(status_train)

from sklearn.preprocessing import MinMaxScaler  # 0~1
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler

scaler = MinMaxScaler()
err_train = scaler.fit_transform(err_train)

def define_discriminator(in_shape=(10,1), n_classes=7):
    # weight initialization
    inputs = tf.keras.Input(shape=(len(err_train[0]),))
    fe = Dense(16)(inputs)
    fe = tf.nn.relu(fe)
    fe = Dense(16)(fe)
    fe = tf.nn.relu(fe)
    # real/fake output
    out1 = Dense(1, activation='sigmoid')(fe)
    # class label output
    out2 = Dense(n_classes, activation='softmax')(fe)
    # define model
    model = Model(inputs, [out1, out2])
    # compile model
    opt = Adam(lr=0.0001, beta_1=0.5)
    model.compile(loss=['binary_crossentropy', 'sparse_categorical_crossentropy'], optimizer=opt)
    return model

def define_generator(latent_dim, n_classes=7):
    in_lat = Input(shape=(latent_dim,))
    n_nodes = 16
    gen = Dense(n_nodes)(in_lat)
    gen = tf.nn.tanh(gen)
    gen = Dense(16)(gen)
    gen = tf.nn.tanh(gen)
    out_layer = Dense(len(err_train[0]), activation='linear')(gen)
    # define model
    model = Model(in_lat, out_layer)
    return model


# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model):
    # make weights in the discriminator not trainable
    d_model.trainable = False
    # connect the outputs of the generator to the inputs of the discriminator
    gan_output = d_model(g_model.output)
    # define gan model as taking noise and label and outputting real/fake and label outputs
    model = Model(g_model.input, gan_output)
    # compile model
    opt = Adam(lr=0.0001, beta_1=0.5)
    model.compile(loss=['binary_crossentropy', 'sparse_categorical_crossentropy'], optimizer=opt)
    return model


# load images
def load_real_samples():
    (trainX, trainy) = (err_train, status_train)
    print(trainX.shape, trainy.shape)
    return [trainX, trainy]


# select real samples
def generate_real_samples(dataset, n_samples):
    # split into images and labels
    images, labels = dataset
    # choose random instances
    ix = randint(0, images.shape[0], n_samples)
    # select images and labels
    X, labels = images[ix], labels[ix]
    # generate class labels
    y = ones((n_samples, 1))
    return [X, labels], y


# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples, n_classes = 7):
    # generate points in the latent space
    x_input = randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    z_input = x_input.reshape(n_samples, latent_dim)
    # generate labels
    labels = randint(0, n_classes-1, n_samples)
    return [z_input, labels]


# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n_samples):
    # generate points in latent space
    z_input, labels_input = generate_latent_points(latent_dim, n_samples)
    # predict outputs
    images = generator.predict([z_input, labels_input])
    # create class labels
    y = zeros((n_samples, 1))
    return [images, labels_input], y


# generate samples and save as a plot and save the model
def summarize_performance(step, g_model, d_model, latent_dim, n_samples=100):

    X, y = dataset
    y_pred = d_model.predict(X)
    y_pred = y_pred[1:]
    y_pred = np.argmax(y_pred, axis=2)

    print(classification_report(y, y_pred[0,:]))

    filename1 = 'model/generated_plot_%04d.png' % (step + 1)
    pyplot.savefig(filename1)
    pyplot.close()
    # save the generator model
    filename2 = 'model/g_model_%04d.h5' % (step + 1)
    g_model.save(filename2)
    filename3 = 'model/d_model_%04d.h5' % (step + 1)
    d_model.save(filename3)
    filename4 = 'model/gan_model_%04d.h5' % (step + 1)
    gan_model.save(filename4)
    print('>Saved: %s, %s, %s and %s' % (filename1, filename2, filename3, filename4))


# train the generator and discriminator

def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=2000, n_batch=40):

    bat_per_epo = int(dataset[0].shape[0] / n_batch)
    # calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs
    # calculate the size of half a batch of samples
    #half_batch = int(n_batch / 2)
    half_batch = n_batch
    # manually enumerate epochs
    for i in range(n_steps):
        # get randomly selected 'real' samples
        [X_real, labels_real], y_real = generate_real_samples(dataset, half_batch)
        # update discriminator model weights
        _, d_r1, d_r2 = d_model.train_on_batch(X_real, [y_real, labels_real])
        # generate 'fake' examples
        [X_fake, labels_fake], y_fake = generate_fake_samples(g_model, latent_dim, half_batch)

        # print('GENERATE_FAKE_SAMPLES')
        # print(X_fake)
        # update discriminator model weights
        _, d_f, d_f2 = d_model.train_on_batch(X_fake, [y_fake, labels_fake])
        # prepare points in latent space as input for the generator
        [z_input, z_labels] = generate_latent_points(latent_dim, n_batch)
        # create inverted labels for the fake samples
        y_gan = ones((n_batch, 1))
        # update the generator via the discriminator's error
        _, g_1, g_2 = gan_model.train_on_batch([z_input, z_labels], [y_gan, z_labels])
        # summarize loss on this batch
        print('>%d, dr[%.3f,%.3f], df[%.3f,%.3f], g[%.3f,%.3f]' % (i + 1, d_r1, d_r2, d_f, d_f2, g_1, g_2))
        # evaluate the model performance every 'epoch'
        if (i + 1) % (bat_per_epo * 20) == 0:
            summarize_performance(i, g_model, d_model, latent_dim)

            X_fake = scaler.inverse_transform(X_fake)
            X_fake_temp = X_fake
            labels_fake_temp = labels_fake
            X_fake_temp = X_fake_temp.reshape(40, 10)
            labels_fake_temp = labels_fake_temp.reshape(40, 1)
            generated_fake_data = np.append(X_fake_temp, labels_fake_temp, axis=1)

            np.savetxt('generated_data/generated_fake_data %s th.csv' % (i + 1), generated_fake_data, delimiter=",")


# size of the latent space
latent_dim = 3
# create the discriminator
discriminator = define_discriminator()
# create the generator
generator = define_generator(latent_dim)
# create the gan
gan_model = define_gan(generator, discriminator)
# load image data
dataset = load_real_samples()
# train model
train(generator, discriminator, gan_model, dataset, latent_dim)