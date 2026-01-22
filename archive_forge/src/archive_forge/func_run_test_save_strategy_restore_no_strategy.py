import os
import numpy as np
import tensorflow.compat.v2 as tf
from absl.testing import parameterized
from keras.src.distribute import model_combinations
def run_test_save_strategy_restore_no_strategy(self, model_and_input, distribution, save_in_scope):
    """Save a model with DS, and restore it without DS."""
    saved_dir = os.path.join(self.get_temp_dir(), '1')
    with distribution.scope():
        model = model_and_input.get_model()
        x_train, y_train, x_predict = model_and_input.get_data()
        batch_size = model_and_input.get_batch_size()
        self._train_model(model, x_train, y_train, batch_size)
        predict_dataset = self._get_predict_dataset(x_predict, batch_size)
        result_before_save = self._predict_with_model(distribution, model, predict_dataset)
    if save_in_scope:
        with distribution.scope():
            self._save_model(model, saved_dir)
    else:
        self._save_model(model, saved_dir)
    load_result = self._load_and_run_model(distribution=None, saved_dir=saved_dir, predict_dataset=predict_dataset)
    self.assertAllClose(result_before_save, load_result)