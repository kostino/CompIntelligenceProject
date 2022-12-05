import os
import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import KFold
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
import seaborn as sn

n_hidden_1 = [64, 128]
n_hidden_2 = [256, 512]
alpha = [0.1, 0.001, 0.000001]
learning_rate = [0.1, 0.01, 0.001]
times = []
k = 5


def precision_m(y_true, y_pred):
    """Precision metric.
    Only computes a batch-wise average of precision.
    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall_m(y_true, y_pred):
    """Recall metric.
        Only computes a batch-wise average of recall.
        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


(trainx, trainy), (testx, testy) = keras.datasets.mnist.load_data()

trainx, valx = tf.split(trainx, [int(48000), int(12000)], 0)
trainy, valy = tf.split(trainy, [int(48000), int(12000)], 0)


trainy = to_categorical(trainy, 10)
valy = to_categorical(valy, 10)


x = np.array(trainx)
y = np.array(trainy)

# grid = []
# f_measures_f = []
# index = 0
# start = time.time()
# train_split = 0.8
# val_split = 0.2
# fold = 0
# for nh1 in n_hidden_1:
#     for nh2 in n_hidden_2:
#         for a in alpha:
#             for lr in learning_rate:
#                 print(f'At this moment n_h1=' + str(nh1) + ' n_h2=' + str(nh2) + ' a=' + str(a) +
#                       ' and lr=' + str(lr))
#                 # Grid Search
#                 f_measures = []
#                 kf = KFold(5, shuffle=True, random_state=4)  # Use for KFold classification with random_state = 42
#                 fold = 0
#                 for train, test in kf.split(x):
#                     fold += 1
#
#                     x_train = x[train]
#                     y_train = y[train]
#                     x_val = x[test]
#                     y_val = y[test]
#
#                     model = keras.models.Sequential()
#                     model.add(keras.layers.Flatten(input_shape=(28, 28)))
#                     model.add(keras.layers.Dense(nh1, activation='relu', kernel_regularizer=l2(a),
#                                                  kernel_initializer=HeNormal()))
#                     model.add(keras.layers.Dense(nh2, activation='relu', kernel_regularizer=l2(a),
#                                                  kernel_initializer=HeNormal()))
#                     model.add(keras.layers.Dense(10, activation='softmax', kernel_regularizer=l2(a),
#                                                  kernel_initializer=HeNormal()))
#                     model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=lr),
#                                   loss=tf.keras.losses.CategoricalCrossentropy(),
#                                   metrics=[f1_m])
#                     history = model.fit(x_train, y_train, validation_split=0.2, batch_size=256, epochs=1000,
#                                         callbacks=[EarlyStopping(monitor='val_f1_m', patience=200, mode="max")])
#                     f_measures.append(max(history.history['val_f1_m']))
#
#                 f_measures_f.append(sum(f_measures) / len(f_measures))
#                 print(f'F_measures is = ' + str(f_measures_f[-1]))
#                 grid.append((nh1, nh2, a, lr))
# times.append((time.time() - start))
# print(f'It lasted ' + str(times[0]) + ' secs')
# if not os.path.exists('kfold'):
#     os.mkdir('kfold')
# with open('kfold' + '/fscores', 'wb') as file_pi:
#     pickle.dump(f_measures_f, file_pi)
# with open('kfold' + '/grid', 'wb') as file_pi:
#     pickle.dump(grid, file_pi)
# with open('kfold' + '/times', 'wb') as file_pi:
#     pickle.dump(times, file_pi)

# max_val_of_f_measure = max(f_measures_f)
# for f_value in range(0, len(f_measures_f)):
#     if f_measures_f[f_value] == max_val_of_f_measure:
#         print(f'Max f_measure is ' + str(f_measures_f[f_value]) +
#               ' with: n_h1=' + str(grid[index][0]) + ' n_h2=' + str(grid[index][1]) +
#               ' a=' + str(grid[index][2]) + ' lr=' + str(grid[index][3]) + '')
#     index += 1

model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=(28, 28)))
model.add(keras.layers.Dense(128, activation='relu', kernel_regularizer=l2(0.001),
                kernel_initializer=HeNormal()))
model.add(keras.layers.Dense(256, activation='relu', kernel_regularizer=l2(0.001),
                kernel_initializer=HeNormal()))
model.add(keras.layers.Dense(10, activation='softmax', kernel_regularizer=l2(0.001),
                kernel_initializer=HeNormal()))
model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=[f1_m, precision_m, recall_m, 'accuracy'])
history = model.fit(trainx, trainy, batch_size=256, epochs=1000, validation_data=(valx, valy),
                    callbacks=[EarlyStopping(monitor='val_f1_m', patience=200, mode="max")])

trn_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

trn_loss = history.history['loss']
val_loss = history.history['val_loss']

trn_f1_m = history.history['f1_m']
val_f1_m = history.history['val_f1_m']

plt.plot(trn_f1_m)
plt.plot(val_f1_m)

plt.title('F1 measure for h_h1=128, n_h2=256, a=0.001, lr=0.001')
plt.xlabel('Epochs')
plt.ylabel('F1 Measure')
plt.legend(['Training F1 measure', 'Validation F1 measure'], loc='lower right')
plt.savefig('kfold/F1_measure_Optimal.png')
plt.close()

plt.plot(trn_loss)
plt.plot(val_loss)

plt.title('Losses for h_h1=128, nh2=256, a=0.001, lr=0.001')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Training Losses', 'Validation Losses'], loc='lower right')
plt.savefig('kfold/Losses_Optimal.png')
plt.close()

plt.plot(trn_acc)
plt.plot(val_acc)

plt.title('Accuracy for h_h1=128, nh2=256, a=0.001, lr=0.001')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['Training Accuracy', 'Validation Accuracy'], loc='lower right')
plt.savefig('kfold/Accuracy_Optimal.png')
plt.close()

predict = model.predict(testx)
con_matrix = confusion_matrix(testy, predict.argmax(axis=1))
sn.heatmap(con_matrix, annot=True, fmt='g')
recalls = [con_matrix[i][i] / sum(con_matrix[i]) for i in range(10)]
precisions = [con_matrix[i][i] / sum([row[i] for row in con_matrix]) for i in range(10)]
