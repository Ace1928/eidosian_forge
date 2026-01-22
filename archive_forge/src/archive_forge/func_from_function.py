from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import six
import tensorflow as tf
from tensorflow.python.eager import function
from tensorflow.python.eager import wrap_function
from tensorflow.python.framework import func_graph
from tensorflow.python.saved_model.model_utils import export_utils
from tensorflow.python.util import function_utils
from tensorflow_estimator.python.estimator import model_fn as model_fn_lib
from tensorflow_estimator.python.estimator.mode_keys import ModeKeys
@staticmethod
def from_function(model_fn, all_modes=None, config=None, params=None):
    """Creates a new ModelFunction object from a model function."""
    if all_modes is None:
        all_modes = [ModeKeys.TRAIN, ModeKeys.EVAL, ModeKeys.PREDICT]
    else:
        all_modes = list(all_modes)
    obj = ModelFunction(config=config, params=params)
    for mode in all_modes:
        obj.add_mode(model_fn, mode)
    return obj