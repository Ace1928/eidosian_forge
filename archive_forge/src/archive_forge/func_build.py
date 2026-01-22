from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import six
import tensorflow as tf
from tensorflow.python.feature_column import feature_column
from tensorflow.python.feature_column import feature_column_lib
from tensorflow.python.feature_column import feature_column_v2 as fc_v2
from tensorflow.python.framework import ops
from tensorflow.python.ops import variable_scope
from tensorflow_estimator.python.estimator import estimator
from tensorflow_estimator.python.estimator.canned import head as head_lib
from tensorflow_estimator.python.estimator.canned import optimizers
from tensorflow_estimator.python.estimator.canned.linear_optimizer.python.utils import sdca_ops
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
from tensorflow_estimator.python.estimator.head import binary_class_head
from tensorflow_estimator.python.estimator.head import head_utils
from tensorflow_estimator.python.estimator.head import regression_head
from tensorflow_estimator.python.estimator.mode_keys import ModeKeys
def build(self, _):
    with variable_scope._pure_variable_scope(self.name):
        for column in self._feature_columns:
            with variable_scope._pure_variable_scope(fc_v2._sanitize_column_name_for_variable_scope(column.name)):
                column.create_state(self._state_manager)
                if isinstance(column, fc_v2.CategoricalColumn):
                    first_dim = column.num_buckets
                else:
                    first_dim = column.variable_shape.num_elements()
                self._state_manager.create_variable(column, name='weights', dtype=tf.float32, shape=(first_dim, self._units), initializer=tf.keras.initializers.zeros(), trainable=self.trainable)
        self.bias = self.add_weight(name='bias_weights', dtype=tf.float32, shape=[self._units], initializer=tf.keras.initializers.zeros(), trainable=self.trainable, use_resource=True, getter=tf.compat.v1.get_variable)
    super(_LinearModelLayer, self).build(None)