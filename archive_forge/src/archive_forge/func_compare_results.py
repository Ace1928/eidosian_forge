import functools
import numpy as np
import tensorflow.compat.v2 as tf
from absl.testing import parameterized
import keras.src as keras
from keras.src.distribute import distributed_training_utils
from keras.src.distribute.strategy_combinations import all_strategies
from keras.src.distribute.strategy_combinations import (
from keras.src.distribute.strategy_combinations import strategies_minus_tpu
from keras.src.mixed_precision import policy
from keras.src.utils import data_utils
def compare_results(results_with_ds, results_without_ds, distribution, testcase, partial_last_batch=None):
    """Compares results of model compiled with/without distribution strategy."""
    if policy.global_policy().compute_dtype in ('float16', 'bfloat16'):
        default_tolerance = 0.01
        relaxed_tolerance = 0.01
    elif partial_last_batch == 'train_and_eval':
        default_tolerance = 0.001
        relaxed_tolerance = 0.001
    else:
        default_tolerance = 4e-05
        relaxed_tolerance = 0.0001

    def _get_compare_result_tolerance(key):
        """Returns tolerance to compare results."""
        if tf.test.is_gpu_available() and key.startswith(('weights_1', 'weights_2', 'predict_result')):
            return relaxed_tolerance
        return default_tolerance
    for key in sorted(results_with_ds.keys()):
        if key.startswith('training_history') and isinstance(distribution, (tf.distribute.experimental.TPUStrategy, tf.compat.v1.distribute.experimental.TPUStrategy)) and (distribution.extended.steps_per_run > 1):
            continue
        tolerance = _get_compare_result_tolerance(key)
        if partial_last_batch is not None:
            if key.startswith('eval_result'):
                results_with_ds[key] = results_with_ds[key][1:]
                results_without_ds[key] = results_without_ds[key][1:]
            if key.startswith('training_history'):
                results_with_ds[key]['val_loss'] = 0
                results_without_ds[key]['val_loss'] = 0
        testcase.assertAllClose(results_with_ds[key], results_without_ds[key], atol=tolerance, rtol=tolerance, msg=f'Fail to assert {key}.')