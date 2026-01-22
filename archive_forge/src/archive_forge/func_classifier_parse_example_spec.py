from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import six
import tensorflow as tf
from tensorflow.python.feature_column import feature_column_lib as fc
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
@estimator_export(v1=['estimator.classifier_parse_example_spec'])
def classifier_parse_example_spec(feature_columns, label_key, label_dtype=tf.dtypes.int64, label_default=None, weight_column=None):
    parsing_spec = tf.compat.v1.feature_column.make_parse_example_spec(feature_columns)
    label_spec = tf.io.FixedLenFeature((1,), label_dtype, label_default)
    return _add_label_and_weight_to_parsing_spec(parsing_spec=parsing_spec, label_key=label_key, label_spec=label_spec, weight_column=weight_column)