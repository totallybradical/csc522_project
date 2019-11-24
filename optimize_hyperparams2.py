#####
#
# TODO: Show the super awesome epoch plot on final model
#
##### 

########
# SEED
########

# Seed for consistent output
from numpy.random import seed
import tensorflow as tf
my_seed = 30
seed(my_seed)
tf.random.set_seed(my_seed)

# Required imports
import glob
import matplotlib.pyplot as plt
import numpy as np
from numpy import loadtxt
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Sequential
from keras.optimizers import adam
from keras.regularizers import l1
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from time import time

# KerasClassifier function
def create_model(dr=0.0, hn=1):
	# Model
	# Setup Layers (Keras Model) with requested parameters
    model = Sequential()
    model.add(Dense(hn, input_dim=1440, activation='relu'))
    model.add(Dropout(dr))
    model.add(Dense(1, activation="sigmoid"))
    # Optimizer
    opt = adam(lr=0.001)
	# Compile model
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

data = []
labels = []

########
# HOT
########

hot_data = []
hot_csv_paths = glob.glob("/home/bradical/Projects/Project/data/hot_train/*.csv")

# Flatten and compile data
for hot_csv_path in hot_csv_paths:
    csv_data = np.genfromtxt(hot_csv_path, delimiter=',')
    csv_flat = np.array(csv_data).flatten().tolist()
    hot_data.append(csv_flat)
    labels.append("hot")

########
# COLD
########

cold_data = []
cold_csv_paths = glob.glob("/home/bradical/Projects/Project/data/cold_train/*.csv")

# Flatten and compile data
for cold_csv_path in cold_csv_paths[:-2]: # Minus 2 so total rows with hot = 330 (easier to split)
    csv_data = np.genfromtxt(cold_csv_path, delimiter=',')
    csv_flat = np.array(csv_data).flatten().tolist()
    cold_data.append(csv_flat)
    labels.append("cold")

########
# COMBINE
########

data = np.concatenate((hot_data, cold_data), axis=0)
min_max_scaler = preprocessing.MinMaxScaler()
data_minmax = min_max_scaler.fit_transform(data)
labels = np.array(labels)

print("data Dimensions: %s" % str(data.shape))
print("labels Dimensions: %s" % str(labels.shape))

########
# BINARIZE
########

binarizer = preprocessing.LabelBinarizer()
labels_bin = binarizer.fit_transform(labels)

########
# SPLIT
########

# 80% 20% 
x_train, x_test, y_train, y_test = train_test_split(data_minmax, labels_bin, test_size=0.2)

print("x_train Dimensions: %s" % str(str(x_train.shape)))
print("x_test Dimensions: %s" % str(str(x_test.shape)))
print("y_train Dimensions: %s" % str(str(y_train.shape)))
print("y_test Dimensions: %s" % str(str(y_test.shape)))

########
# MODEL
########

# batch_size = 248
# batch_size = list(range(24, 288, 24))
batch_size = 96

# epochs = 475
# epochs = list(range(50, 550, 50))
epochs = 300
# param_grid = dict(batch_size=batch_size, epochs=epochs)

### Learning Rate ###
# lr = [0.001, 0.005, 0.01]
lr = [0.001]
# dr = [0.0, 0.1, 0.2]
dr = 0.1
# hn = [1, 5, 10, 15, 20, 25, 30]
hn = 20

model = KerasClassifier(build_fn=create_model, verbose=20, batch_size=batch_size, epochs=epochs)

########
# GRID SEARCH
########

print()

param_grid = dict(dr=dr, hn=hn)

num_folds = 10
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=num_folds)
start_time = time()
grid_result = grid.fit(x_train, y_train)
print("Completed in %0.3fs" % (time() - start_time))
print()

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))





# x_train, x_validate, x_test = np.split(data_minmax, [int(.6*len(data_minmax)), int(.8*len(data_minmax))])
# y_train, y_validate, y_test = np.split(labels_bin, [int(.6*len(labels_bin)), int(.8*len(labels_bin))])

# # Model + Training
# # hidden_neurons = [2, 15, 30, 50, 65, 350, 450, 650, 1292, 1400]
# # Range to 200, jump at 5
# hidden_neurons = range(20)
# accuracy_by_neurons = {}
# # Loop through each hidden neuron value
# for hn in hidden_neurons:
#     print()
#     print("-------------------")
#     print("HIDDEN NEURON: %d" % hn)
#     print("-------------------")
#     print()
#     # Setup Layers (Keras Model) with requested parameters
#     model = Sequential()
#     model.add(Dense(hn, input_dim=data.shape[1], activation='relu'))
#     model.add(Dense(1, activation="sigmoid"))

#     # lrs = [0.0001, 0.001, 0.01, 0.1]
#     lrs = [0.001]

#     for lr in lrs:
#         opt = adam(lr=lr)

#         model.compile(
#             loss="binary_crossentropy",
#             optimizer=opt,
#             metrics=["accuracy"]
#         )

#         # Training constants
#         EPOCHS = 50 # One pass through all of the rows in the training dataset.
#         BATCH_SIZE = 99 # One or more samples considered by the model within an epoch before weights are updated.

#         # Train the model with requested parameters
#         training = model.fit(
#             x_train, 
#             y_train, 
#             validation_data=(x_validate, y_validate),
#             epochs=EPOCHS, 
#             batch_size=BATCH_SIZE
#         )
#         print()
        
#         # Print out + store training and validation accuracy
#         training_accuracy = training.history['accuracy'][-1]
#         print("Training Accuracy: %.2f" % training_accuracy)
#         validation_accuracy = training.history['val_accuracy'][-1]
#         print("Validation Accuracy: %.2f" % validation_accuracy)
#         accuracy_by_neurons[hn] = [training_accuracy, validation_accuracy]
#         print()

#         # Evaluate with Test Data
#         _, accuracy = model.evaluate(x_test, y_test)
#         print('Testing Accuracy: %.2f' % (accuracy*100))
#         print()

# # Plot model accuracy
# neuron_values = list(accuracy_by_neurons.keys())
# training_accuracies = [acc[0] for acc in accuracy_by_neurons.values()]
# validation_accuracies = [acc[1] for acc in accuracy_by_neurons.values()]
# plt.plot(neuron_values, training_accuracies, color='b')
# plt.plot(neuron_values, validation_accuracies, color='g')
# plt.title('Model Accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Hidden Neurons')
# plt.legend(['training', 'validation'], loc='upper left')
# plt.show()
