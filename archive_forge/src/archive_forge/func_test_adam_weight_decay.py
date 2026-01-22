import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from autokeras import keras_layers as layer_module
def test_adam_weight_decay(tmp_path):
    model = keras.Sequential([keras.layers.Dense(10, input_shape=(10,))])
    lr_schedule = keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=0.1, decay_steps=100, end_learning_rate=0.0)
    lr_schedule = layer_module.WarmUp(initial_learning_rate=0.1, decay_schedule_fn=lr_schedule, warmup_steps=10)
    optimizer = layer_module.AdamWeightDecay(learning_rate=lr_schedule, weight_decay_rate=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-06, exclude_from_weight_decay=['LayerNorm', 'layer_norm', 'bias'])
    model.compile(loss='mse', optimizer=optimizer)
    model.fit(np.random.rand(100, 10), np.random.rand(100, 10), epochs=2)
    model.save(os.path.join(tmp_path, 'model'))