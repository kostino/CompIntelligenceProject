from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import boston_housing
from tensorflow.keras import layers
from tensorflow.keras.initializers import Initializer
from sklearn.cluster import KMeans
import pickle

# Useful link -> https://github.com/PetraVidnerova/rbf_keras


class RBFLayer(layers.Layer):
    # output_dim: number of hidden units (number of outputs of the layer)
    # initializer: instance of initializer to initialize centers
    # betas: float, initial value for betas (beta = 1 / 2*pow(sigma,2))
    def __init__(self, output_dim, initializer, betas=1.0, **kwargs):
        self.betas = betas
        self.output_dim = output_dim
        self.initializer = initializer
        super(RBFLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.centers = self.add_weight(name='centers', shape=(self.output_dim, input_shape[1]),
                                       initializer=self.initializer, trainable=False)
        d_max = 0
        for i in range(0, self.output_dim):
            for j in range(0, self.output_dim):
                d = np.linalg.norm(self.centers[i] - self.centers[j])
                if d > d_max:
                    d_max = d
        sigma = d_max / np.sqrt(2 * self.output_dim)
        self.betas = np.ones(self.output_dim) / (2 * (sigma ** 2))
        super(RBFLayer, self).build(input_shape)

    def call(self, inputs, *args, **kwargs):
        C = tf.expand_dims(self.centers, -1)  # εισάγουμε μια διάσταση από άσσους
        H = tf.transpose(C - tf.transpose(inputs))  # Πίνακας με τις διαφορές
        return tf.exp(-self.betas * tf.math.reduce_sum(H ** 2, axis=1))

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.output_dim


class InitCentersKMeans(Initializer):
    def __init__(self, X, max_iter=100):
        self.X = X
        self.max_iter = max_iter
        super().__init__()

    def __call__(self, shape, dtype=None, *args):
        assert shape[1] == self.X.shape[1]
        n_centers = shape[0]
        km = KMeans(n_clusters=n_centers, max_iter=self.max_iter, verbose=0)
        km.fit(self.X)
        return km.cluster_centers_


def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


def mse(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true))


def r2_score(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res / (SS_tot + K.epsilon())


(trainx, trainy), (testx, testy) = boston_housing.load_data(test_split=0.25)

#normalize
trainx = (trainx - np.mean(trainx, axis=0)) / np.std(trainx, axis=0)
testx = (testx - np.mean(testx, axis=0)) / np.std(testx, axis=0)



trainx, valx = tf.split(trainx, [int(trainx.shape[0] * 0.8), trainx.shape[0] - int(trainx.shape[0] * 0.8)], 0)
trainy, valy = tf.split(trainy, [int(trainy.shape[0] * 0.8), trainy.shape[0] - int(trainy.shape[0] * 0.8)], 0)

neuron_1 = 0.1 * trainy.shape[0]
neuron_2 = 0.5 * trainy.shape[0]
neuron_3 = 0.9 * trainy.shape[0]

neurons = [neuron_1, neuron_2, neuron_3]
i = 1
types_of_neurons = ["10%", "50%", "90%"]
for i, n in enumerate(neurons):
    model = Sequential()
    model.add(RBFLayer(int(n), initializer=InitCentersKMeans(trainx), input_shape=(13,)))
    model.add(Dense(128))
    model.add(Dense(1))

    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001), loss=mse, metrics=[r2_score, rmse])
    history = model.fit(trainx, trainy, batch_size=4, epochs=100, validation_data=(valx, valy))
    score = model.evaluate(testx, testy)

    with open('trainHistoryDictRBF' + types_of_neurons[i-1], 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    train_r2 = history.history['r2_score']
    val_r2 = history.history['val_r2_score']

    train_rmse = history.history['rmse']
    val_rmse = history.history['val_rmse']

    plt.plot(train_loss)
    plt.plot(val_loss)

    plt.title('Learning Curve for ' + str(types_of_neurons[i - 1]) + ' of training dataset and lr=0.001')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(['Training Loss', 'Validation Loss'], loc='upper right')
    plt.savefig(str(types_of_neurons[i - 1]) + '_loss.png')
    plt.close()

    plt.plot(train_r2)
    plt.plot(val_r2)

    plt.title('R2 for ' + str(types_of_neurons[i - 1]) + ' of training dataset and lr=0.001')
    plt.xlabel('Epochs')
    plt.ylabel('R2')
    plt.legend(['Training R2', 'Validation R2'], loc='lower right')
    plt.savefig(str(types_of_neurons[i - 1]) + '_R2.png')
    plt.close()

    plt.plot(train_rmse)
    plt.plot(val_rmse)

    plt.title('RMSE for ' + str(types_of_neurons[i - 1]) + ' of training dataset and lr=0.001')
    plt.xlabel('Epochs')
    plt.ylabel('RMSE')
    plt.legend(['Training RMSE', 'Validation RMSE'], loc='lower right')
    plt.savefig(str(types_of_neurons[i - 1]) + '_RMSE.png')
    plt.close()

