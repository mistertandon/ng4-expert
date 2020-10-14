import matplotlib.pyplot as plt


import numpy as np

import kerastuner as kt

np.random.seed(42)

import tensorflow as tf

from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from kerastuner.tuners import RandomSearch



(D_Train, Y_Train), (D_Test, Y_Test) = fashion_mnist.load_data()

font_dict_params = {'family': 'serif', 'color': '#080808', 'weight': 'normal', 'size': 14}
figure_inst_i = plt.figure(figsize=(10, 4))
for i in range(24):
    plt.subplot(3, 8, i + 1)
    plt.imshow(D_Train[i])
    plt.xticks([])
    plt.yticks([])

    plt.xlabel('yup', fontdict=font_dict_params)
    plt.grid(True)

plt.show()
figure_inst_i.tight_layout()

input_pixels = np.multiply(D_Train.shape[1], D_Train.shape[2])
d_train_raw = D_Train.reshape(D_Train.shape[0], input_pixels).astype('float32') / D_Train.max()
d_test_raw = D_Test.reshape(D_Test.shape[0], input_pixels).astype('float32') / D_Train.max()

y_train_raw = Y_Train.astype('float32')

BATCH_SIZE = 32

def build_model(hp):

    model = Sequential()

    model.add(
        Flatten(input_shape = (28, 28))
    )

    hp_units = hp.Int(
        name = 'units',
        min_value = 32,
        max_value = 512,
        step = 32
    )

    model.add(Dense(units = hp_units, activation = 'relu'))

    model.add(Dense(units = 10))

    hp_learning_rate = hp.Choice(
        'learning_rate',
        values = [1e-2, 1e-3, 1e-4]
    )

    model.compile(
        optimizer = Adam(lr = hp_learning_rate),
        loss = SparseCategoricalCrossentropy(from_logits = True),
        metrics = ['accuracy']
    )

    return model


kt.Hyperband(
    build_model,
    objective = 'val_accuracy',
    max_epochs = 5,
    factor = 3,
    seed = 6,
    directory = 'my_dir',
    project_name = 'intro_to_kt'

)

tuner = RandomSearch(
        build_model,
        objective = 'val_accuracy',
        seed = 6,
        max_trials = 10,
        executions_per_trial = 3,
        directory = 'my_dir',
        project_name = 'kt_demo'

)

tuner.search_space_summary(extended = True)

tuner.search(x = D_Train, y = Y_Train, epochs = 5, validation_data = (D_Test, Y_Test))

best_models = tuner.get_best_models(num_models=2)
best_hps = tuner.get_best_hyperparameters(num_trials = 1)

tuner.results_summary()