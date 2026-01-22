from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import os
import shutil
import tempfile
import numpy as np
import six
import tensorflow as tf
from tensorflow.core.example import example_pb2
from tensorflow.core.example import feature_pb2
from tensorflow.python.feature_column import feature_column
from tensorflow.python.feature_column import feature_column_v2
from tensorflow.python.framework import ops
from tensorflow.python.ops import variables as variables_lib
from tensorflow_estimator.python.estimator import estimator
from tensorflow_estimator.python.estimator import run_config
from tensorflow_estimator.python.estimator.canned import linear
from tensorflow_estimator.python.estimator.canned import metric_keys
from tensorflow_estimator.python.estimator.export import export
from tensorflow_estimator.python.estimator.inputs import numpy_io
from tensorflow_estimator.python.estimator.inputs import pandas_io
def begin(self):
    with tf.compat.v1.variable_scope(tf.compat.v1.get_variable_scope()) as scope:
        scope.reuse_variables()
        partitioned_weight = tf.compat.v1.get_variable(self._var_name, shape=(self._var_dim, 1))
        self._test_case.assertTrue(isinstance(partitioned_weight, variables_lib.PartitionedVariable))
        for part in partitioned_weight:
            self._test_case.assertEqual(self._var_dim // self._partitions, part.get_shape()[0])