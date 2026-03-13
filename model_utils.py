## model utils


import tensorflow as tf
import numpy as np


from data_utils import Normalize

def load_model(model_name):
    model = tf.keras.models.load_model(model_name)
    model.summary()
    return model

def predict_ferrite_fraction(X_data, model_name, apply_normalization=True):
    '''
    Here the X_data should already be normalized to 1000
    '''
    model = load_model(model_name)
    X = X_data.reshape(X_data.shape[0], int(X_data.shape[1] / 2), 2)
    if apply_normalization:
        X = Normalize(X)
    Y = model.predict(X).flatten()
    Y = np.around(Y, decimals=3)
    return Y


def predict_ferrite_fraction_batched(X_data, model_name, batch_size=512, apply_normalization=True):
    '''
    Predict in chunks to reduce memory usage when stacking many profiles.
    '''
    model = load_model(model_name)
    results = []
    total = X_data.shape[0]
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        X_batch = X_data[start:end]
        X = X_batch.reshape(X_batch.shape[0], int(X_batch.shape[1] / 2), 2)
        if apply_normalization:
            X = Normalize(X)
        Y = model.predict(X).flatten()
        results.append(Y)
    Y_all = np.concatenate(results)
    return np.around(Y_all, decimals=3)
