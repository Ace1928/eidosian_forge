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
def get_train_step(self, state_manager, weight_column_name, loss_type, feature_columns, features, targets, bias_var, global_step):
    """Returns the training operation of an SdcaModel optimizer."""
    batch_size = tf.compat.v1.shape(targets)[0]
    cache = tf.compat.v2.__internal__.feature_column.FeatureTransformationCache(features)
    dense_features, dense_feature_weights = ([], [])
    sparse_feature_with_values, sparse_feature_with_values_weights = ([], [])
    for column in sorted(feature_columns, key=lambda x: x.name):
        if isinstance(column, feature_column_lib.CategoricalColumn):
            id_weight_pair = column.get_sparse_tensors(cache, state_manager)
            sparse_feature_with_values.append(self._prune_and_unique_sparse_ids(id_weight_pair))
            sparse_feature_with_values_weights.append(state_manager.get_variable(column, 'weights'))
        elif isinstance(column, tf.compat.v2.__internal__.feature_column.DenseColumn):
            if column.variable_shape.ndims != 1:
                raise ValueError('Column %s has rank %d, larger than 1.' % (type(column).__name__, column.variable_shape.ndims))
            dense_features.append(column.get_dense_tensor(cache, state_manager))
            dense_feature_weights.append(state_manager.get_variable(column, 'weights'))
        else:
            raise ValueError('LinearSDCA does not support column type %s.' % type(column).__name__)
    dense_features.append(tf.ones([batch_size, 1]))
    dense_feature_weights.append(bias_var)
    example_weights = tf.reshape(features[weight_column_name], shape=[-1]) if weight_column_name else tf.ones([batch_size])
    example_ids = features[self._example_id_column]
    training_examples = dict(sparse_features=sparse_feature_with_values, dense_features=dense_features, example_labels=tf.compat.v1.to_float(tf.reshape(targets, shape=[-1])), example_weights=example_weights, example_ids=example_ids)
    training_variables = dict(sparse_features_weights=sparse_feature_with_values_weights, dense_features_weights=dense_feature_weights)
    sdca_model = sdca_ops._SDCAModel(examples=training_examples, variables=training_variables, options=dict(symmetric_l1_regularization=self._symmetric_l1_regularization, symmetric_l2_regularization=self._symmetric_l2_regularization, adaptive=self._adaptive, num_loss_partitions=self._num_loss_partitions, num_table_shards=self._num_table_shards, loss_type=loss_type))
    train_op = sdca_model.minimize(global_step=global_step)
    return (sdca_model, train_op)