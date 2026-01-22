from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import abc
import six
import tensorflow as tf
from tensorflow.python.feature_column import feature_column_lib
from tensorflow.python.feature_column.feature_column import _LazyBuilder
from tensorflow.python.feature_column.feature_column import _NumericColumn
from tensorflow.python.framework import ops
from tensorflow.python.util import function_utils
from tensorflow_estimator.python.estimator.canned import metric_keys
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
from tensorflow_estimator.python.estimator.export import export_output
def validate_trainable_variables(trainable_variables=None):
    if trainable_variables is None:
        raise ValueError('trainable_variables cannot be None. Given {}'.format(trainable_variables))
    if not isinstance(trainable_variables, (list, tuple)):
        raise ValueError(_VALIDATION_ERROR_MSG.format('trainable_variables', type(trainable_variables)))