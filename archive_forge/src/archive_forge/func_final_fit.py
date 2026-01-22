import collections
import copy
import os
import keras_tuner
from tensorflow import keras
from tensorflow import nest
from tensorflow.keras import callbacks as tf_callbacks
from tensorflow.keras.layers.experimental import preprocessing
from autokeras import pipeline as pipeline_module
from autokeras.utils import data_utils
from autokeras.utils import utils
def final_fit(self, **kwargs):
    best_trial = self.oracle.get_best_trials(1)[0]
    best_hp = best_trial.hyperparameters
    pipeline, kwargs['x'], kwargs['validation_data'] = self._prepare_model_build(best_hp, **kwargs)
    model = self._build_best_model()
    self.adapt(model, kwargs['x'])
    model, history = utils.fit_with_adaptive_batch_size(model, self.hypermodel.batch_size, **kwargs)
    return (pipeline, model, history)