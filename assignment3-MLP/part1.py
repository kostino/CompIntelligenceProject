import tensorflow as tf
import numpy as np  # Utilities
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow import keras
from tensorflow.keras.regularizers import l1, l2
import time
TRAIN_SET_SIZE = 60000
epoch_num = 100
val_split = 0.2
import pickle
import os

(trainx, trainy), (testx, testy) = keras.datasets.mnist.load_data()

trainx, valx = tf.split(trainx, [int(48000), int(12000)], 0)
trainy, valy = tf.split(trainy, [int(48000), int(12000)], 0)


trainy = to_categorical(trainy, 10)
valy = to_categorical(valy, 10)


# # First experiment
# # batch_sizes = [1, 256, TRAIN_SET_SIZE]
# batch_sizes = [256, TRAIN_SET_SIZE]
# for b1 in batch_sizes:
#     start = time.time()
#     print("At this moment batch size is ", b1, "")
#     model1 = keras.models.Sequential()
#     model1.add(keras.layers.Flatten(input_shape=(28, 28)))
#     model1.add(keras.layers.Dense(128, activation='relu'))
#     model1.add(keras.layers.Dense(256, activation='relu'))
#     model1.add(keras.layers.Dense(10, activation='softmax'))
#     model1.compile(loss=keras.losses.CategoricalCrossentropy(), metrics=[keras.metrics.CategoricalAccuracy()])
#     history1 = model1.fit(trainx, trainy, batch_size=b1, epochs=epoch_num, validation_data=(valx, valy))
#
#     end = time.time()
#
#     trn_acc = [history1.history['categorical_accuracy'][i] * 100 for i in range(100)]
#     val_acc = [history1.history['val_categorical_accuracy'][i] * 100 for i in range(100)]
#     trn_los = history1.history['loss']
#     val_los = history1.history['val_loss']
#
#     plt.plot(trn_acc)
#     plt.plot(val_acc)
#
#     plt.title(f'Accuracy with batch =' + str(b1))
#     plt.xlabel('Epochs')
#     plt.ylabel('Accuracy')
#     plt.legend(['Training Accuracy', 'Validation Accuracy'], loc='lower right')
#     plt.savefig(
#         'Accuracy with batch =' + str(b1) + '.png')
#     plt.close()
#
#     plt.plot(trn_los)
#     plt.plot(val_los)
#
#     plt.title(f'Losses with batch =' + str(b1))
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.legend(['Training Losses', 'Validation Accuracy'], loc='upper right')
#     plt.savefig(
#         'Losses with batch =' + str(b1) + '.png')
#     plt.close()
#     with open('trainHistoryDict' + str(b1), 'wb') as file_pi:
#         pickle.dump(history1.history, file_pi)
#
#     print("Time for batch = " + str(b1) + " is " + str(end - start) + " sec")
#
# #
# for rho in [0.01, 0.99]:
#     print("At this moment rho is ", rho, "")
#     model1 = keras.models.Sequential()
#     model1.add(keras.layers.Flatten(input_shape=(28, 28)))
#     model1.add(keras.layers.Dense(128, activation='relu'))
#     model1.add(keras.layers.Dense(256, activation='relu'))
#     model1.add(keras.layers.Dense(10, activation='softmax'))
#     model1.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001, rho=rho),
#                    loss=tf.keras.losses.CategoricalCrossentropy(),
#                    metrics='accuracy')
#     history1 = model1.fit(trainx, trainy, batch_size=256, epochs=epoch_num, validation_data=(valx, valy))
#     if not os.path.exists('part2rmsprop-rho'+str(rho)):
#         os.mkdir('part2rmsprop-rho'+str(rho))
#     with open('part2rmsprop-rho'+str(rho)+'/trainHistoryDictRMSPROP', 'wb') as file_pi:
#         pickle.dump(history1.history, file_pi)
#
#     plt.plot(history1.history['accuracy'])
#     plt.plot(history1.history['val_accuracy'])
#
#     plt.title(f'Accuracy with rho= ' + str(rho))
#     plt.xlabel('Epochs')
#     plt.ylabel('Accuracy')
#     plt.legend(['Training Accuracy', 'Validation Accuracy'], loc='lower right')
#     plt.savefig('part2rmsprop-rho'+str(rho)+'/accuracy.png')
#     plt.close()
#
#     plt.plot(history1.history['loss'])
#     plt.plot(history1.history['val_loss'])
#
#     plt.title(f'Losses with rho= ' + str(rho))
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.legend(['Training Losses', 'Validation Losses'], loc='upper right')
#     plt.savefig('part2rmsprop-rho'+str(rho)+'/loss.png')
#     plt.close()


#
# print("At this moment regularization choice is lr=0.01 with W = 10")
# model2 = keras.models.Sequential()
# model2.add(keras.layers.Flatten(input_shape=(28, 28)))
# model2.add(keras.layers.Dense(128, activation='relu', kernel_initializer=keras.initializers.RandomNormal(mean=10.0)))
# model2.add(keras.layers.Dense(256, activation='relu', kernel_initializer=keras.initializers.RandomNormal(mean=10.0)))
# model2.add(keras.layers.Dense(10, activation='softmax', kernel_initializer=keras.initializers.RandomNormal(mean=10.0)))
# model2.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
#                loss=tf.keras.losses.CategoricalCrossentropy(),
#                metrics='accuracy')
# history2 = model2.fit(trainx, trainy, batch_size=256, epochs=epoch_num, validation_data=(valx, valy))
#
# if not os.path.exists('part3init'):
#     os.mkdir('part3init')
# with open('part3init/'+'trainHistoryDictSGD', 'wb') as file_pi:
#     pickle.dump(history2.history, file_pi)
# plt.plot(history2.history['accuracy'])
# plt.plot(history2.history['val_accuracy'])
# plt.title(f'(SGD) Accuracy with lr=0.01 with W = 10')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend(['Training Accuracy', 'Validation Accuracy'], loc='lower right')
# plt.savefig('part3init/'+ 'accuracy.png')
# plt.close()
#
# plt.plot(history2.history['loss'])
# plt.plot(history2.history['val_loss'])
# plt.title(f'(SGD) Losses with lr=0.01 with W = 10')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend(['Training Losses', 'Validation Losses'], loc='upper right')
# plt.savefig('part3init/'+ 'loss.png')
# plt.close()

# #
# for reg2 in [0.1, 0.01, 0.001]:
#     print("At this moment regularization choice is l2", " with a = ", reg2, "")
#     model2 = keras.models.Sequential()
#     model2.add(keras.layers.Flatten(input_shape=(28, 28)))
#     model2.add(keras.layers.Dense(128, activation='relu', kernel_regularizer=l2(reg2)))
#     model2.add(keras.layers.Dense(256, activation='relu', kernel_regularizer=l2(reg2)))
#     model2.add(keras.layers.Dense(10, activation='softmax', kernel_regularizer=l2(reg2)))
#     model2.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
#                    loss=tf.keras.losses.CategoricalCrossentropy(),
#                    metrics='accuracy')
#     history2 = model2.fit(trainx, trainy, batch_size=256, epochs=epoch_num, validation_data=(valx, valy))
#
#     if not os.path.exists('part4-LR0.001-l2-' + str(reg2)):
#         os.mkdir('part4-LR0.001-l2-' + str(reg2))
#     with open('part4-LR0.001-l2-' + str(reg2) + '/trainHistoryDictSGD', 'wb') as file_pi:
#         pickle.dump(history2.history, file_pi)
#     plt.plot(history2.history['accuracy'])
#     plt.plot(history2.history['val_accuracy'])
#     plt.title(f'(SGD) Accuracy with lr=0.01 with W = 10')
#     plt.xlabel('Epochs')
#     plt.ylabel('Accuracy')
#     plt.legend(['Training Accuracy', 'Validation Accuracy'], loc='lower right')
#     plt.savefig('part4-LR0.001-l2-' + str(reg2) + '/accuracy.png')
#     plt.close()
#
#     plt.plot(history2.history['loss'])
#     plt.plot(history2.history['val_loss'])
#     plt.title(f'(SGD) Losses with lr=0.01 with W = 10')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.legend(['Training Losses', 'Validation Losses'], loc='upper right')
#     plt.savefig('part4-LR0.001-l2-' + str(reg2) + '/loss.png')
#     plt.close()


print("At this moment regularization choice is l1-norm with a = 0.01")
model2 = keras.models.Sequential()
model2.add(keras.layers.Flatten(input_shape=(28, 28)))
model2.add(keras.layers.Dense(128, activation='relu', kernel_regularizer=l1(0.01)))
model2.add(keras.layers.Dropout(0.3))
model2.add(keras.layers.Dense(256, activation='relu', kernel_regularizer=l1(0.01)))
model2.add(keras.layers.Dropout(0.3))
model2.add(keras.layers.Dense(10, activation='softmax', kernel_regularizer=l1(0.01)))
model2.compile(optimizer=tf.keras.optimizers.SGD(), loss=tf.keras.losses.CategoricalCrossentropy(),
               metrics='accuracy')
history2 = model2.fit(trainx, trainy, batch_size=256, epochs=epoch_num, validation_data=(valx, valy))

with open('trainHistoryDictL1', 'wb') as file_pi:
    pickle.dump(history2.history, file_pi)

if not os.path.exists('part5l1'):
    os.mkdir('part5l1')
with open('part5l1' + '/trainHistoryDictSGD', 'wb') as file_pi:
    pickle.dump(history2.history, file_pi)
plt.plot(history2.history['accuracy'])
plt.plot(history2.history['val_accuracy'])
plt.title(f'(SGD) Accuracy with lr=0.01 with W = 10')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['Training Accuracy', 'Validation Accuracy'], loc='lower right')
plt.savefig('part5l1'+ '/accuracy.png')
plt.close()

plt.plot(history2.history['loss'])
plt.plot(history2.history['val_loss'])
plt.title(f'(SGD) Losses with lr=0.01 with W = 10')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Training Losses', 'Validation Losses'], loc='upper right')
plt.savefig('part5l1'+ '/loss.png')
plt.close()

