from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools
import tensorflow as tf
from tensorflow_estimator.python.estimator import estimator_lib
from tensorflow_estimator.python.estimator.canned import optimizers
from tensorflow_estimator.python.estimator.canned.timeseries import ar_model
from tensorflow_estimator.python.estimator.canned.timeseries import feature_keys
from tensorflow_estimator.python.estimator.canned.timeseries import head as ts_head_lib
from tensorflow_estimator.python.estimator.canned.timeseries import math_utils
from tensorflow_estimator.python.estimator.canned.timeseries import state_management
from tensorflow_estimator.python.estimator.export import export_lib
def build_raw_serving_input_receiver_fn(self, default_batch_size=None, default_series_length=None):
    """Build an input_receiver_fn for export_saved_model which accepts arrays.

    Automatically creates placeholders for exogenous `FeatureColumn`s passed to
    the model.

    Args:
      default_batch_size: If specified, must be a scalar integer. Sets the batch
        size in the static shape information of all feature Tensors, which means
        only this batch size will be accepted by the exported model. If None
        (default), static shape information for batch sizes is omitted.
      default_series_length: If specified, must be a scalar integer. Sets the
        series length in the static shape information of all feature Tensors,
        which means only this series length will be accepted by the exported
        model. If None (default), static shape information for series length is
        omitted.

    Returns:
      An input_receiver_fn which may be passed to the Estimator's
      export_saved_model.
    """

    def _serving_input_receiver_fn():
        """A receiver function to be passed to export_saved_model."""
        placeholders = {}
        time_placeholder = tf.compat.v1.placeholder(name=feature_keys.TrainEvalFeatures.TIMES, dtype=tf.dtypes.int64, shape=[default_batch_size, default_series_length])
        placeholders[feature_keys.TrainEvalFeatures.TIMES] = time_placeholder
        placeholders[feature_keys.TrainEvalFeatures.VALUES] = tf.compat.v1.placeholder_with_default(name=feature_keys.TrainEvalFeatures.VALUES, input=tf.zeros(shape=[default_batch_size if default_batch_size else 0, default_series_length if default_series_length else 0, self._model.num_features], dtype=self._model.dtype), shape=(default_batch_size, default_series_length, self._model.num_features))
        if self._model.exogenous_feature_columns:
            with tf.Graph().as_default():
                parsed_features = tf.compat.v1.feature_column.make_parse_example_spec(self._model.exogenous_feature_columns)
                placeholder_features = tf.compat.v1.io.parse_example(serialized=tf.compat.v1.placeholder(shape=[None], dtype=tf.dtypes.string), features=parsed_features)
                exogenous_feature_shapes = {key: (value.get_shape(), value.dtype) for key, value in placeholder_features.items()}
            for feature_key, (batch_only_feature_shape, value_dtype) in exogenous_feature_shapes.items():
                batch_only_feature_shape = batch_only_feature_shape.with_rank_at_least(1).as_list()
                feature_shape = [default_batch_size, default_series_length] + batch_only_feature_shape[1:]
                placeholders[feature_key] = tf.compat.v1.placeholder(dtype=value_dtype, name=feature_key, shape=feature_shape)
        batch_size_tensor = tf.compat.v1.shape(time_placeholder)[0]
        placeholders.update(self._model_start_state_placeholders(batch_size_tensor, static_batch_size=default_batch_size))
        return export_lib.ServingInputReceiver(placeholders, placeholders)
    return _serving_input_receiver_fn