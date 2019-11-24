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

x_train, x_validate, x_test = np.split(data_minmax, [int(.6*len(data_minmax)), int(.8*len(data_minmax))])
y_train, y_validate, y_test = np.split(labels_bin, [int(.6*len(labels_bin)), int(.8*len(labels_bin))])

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
# epochs = 300
epochs = 100
# param_grid = dict(batch_size=batch_size, epochs=epochs)

### Learning Rate ###
# lr = [0.001, 0.005, 0.01]
lr = [0.001]
# dr = [0.0, 0.1, 0.2]
dr = 0.1
# hn = [1, 5, 10, 15, 20, 25, 30]
hn = 20

# Setup Layers (Keras Model) with requested parameters
model = Sequential()
model.add(Dense(20, input_dim=1440, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1, activation="sigmoid"))
# Optimizer
opt = adam(lr=0.001)
# Compile model
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

# Model + Training        
# # Train the model with requested parameters
training = model.fit(
    x_train, 
    y_train, 
    validation_data=(x_validate, y_validate),
    epochs=epochs, 
    batch_size=batch_size
)
print()
        
# Print out + store training and validation accuracy
training_accuracy = training.history['accuracy'][-1]
print("Training Accuracy: %.2f" % training_accuracy)
validation_accuracy = training.history['val_accuracy'][-1]
print("Validation Accuracy: %.2f" % validation_accuracy)
print()

# Evaluate with Test Data
_, accuracy = model.evaluate(x_test, y_test)
print('Testing Accuracy: %.2f' % (accuracy*100))
print()

# summarize training for accuracy
plt.plot(training.history['accuracy'])
plt.plot(training.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(training.history['loss'])
plt.plot(training.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
