from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import six
import tensorflow as tf
from tensorflow.python.saved_model import model_utils as export_utils
from tensorflow.python.tpu import tensor_tracer
from tensorflow.python.util import function_utils
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
from tensorflow_estimator.python.estimator.mode_keys import ModeKeys
def verify_model_fn_args(model_fn, params):
    """Verifies `model_fn` arguments."""
    args = set(function_utils.fn_args(model_fn))
    if 'features' not in args:
        raise ValueError('model_fn (%s) must include features argument.' % model_fn)
    if params is not None and 'params' not in args:
        raise ValueError('model_fn (%s) does not include params argument, but params (%s) is passed to Estimator.' % (model_fn, params))
    if params is None and 'params' in args:
        tf.compat.v1.logging.warn("Estimator's model_fn (%s) includes params argument, but params are not passed to Estimator.", model_fn)
    non_valid_args = list(args - _VALID_MODEL_FN_ARGS)
    if non_valid_args:
        raise ValueError('model_fn (%s) has following not expected args: %s' % (model_fn, non_valid_args))