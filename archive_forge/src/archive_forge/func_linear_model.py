import abc
import collections
import math
import numpy as np
import six
from tensorflow.python.eager import context
from tensorflow.python.feature_column import utils as fc_utils
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor as sparse_tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import template
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import checkpoint_utils
from tensorflow.python.util import deprecation
from tensorflow.python.util import nest
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
from tensorflow.tools.docs import doc_controls
@doc_controls.header(_FEATURE_COLUMN_DEPRECATION_WARNING)
@tf_export(v1=['feature_column.linear_model'])
@deprecation.deprecated(None, _FEATURE_COLUMN_DEPRECATION_RUNTIME_WARNING)
def linear_model(features, feature_columns, units=1, sparse_combiner='sum', weight_collections=None, trainable=True, cols_to_vars=None):
    """Returns a linear prediction `Tensor` based on given `feature_columns`.

  This function generates a weighted sum based on output dimension `units`.
  Weighted sum refers to logits in classification problems. It refers to the
  prediction itself for linear regression problems.

  Note on supported columns: `linear_model` treats categorical columns as
  `indicator_column`s. To be specific, assume the input as `SparseTensor` looks
  like:

  ```python
    shape = [2, 2]
    {
        [0, 0]: "a"
        [1, 0]: "b"
        [1, 1]: "c"
    }
  ```
  `linear_model` assigns weights for the presence of "a", "b", "c' implicitly,
  just like `indicator_column`, while `input_layer` explicitly requires wrapping
  each of categorical columns with an `embedding_column` or an
  `indicator_column`.

  Example of usage:

  ```python
  price = numeric_column('price')
  price_buckets = bucketized_column(price, boundaries=[0., 10., 100., 1000.])
  keywords = categorical_column_with_hash_bucket("keywords", 10K)
  keywords_price = crossed_column('keywords', price_buckets, ...)
  columns = [price_buckets, keywords, keywords_price ...]
  features = tf.io.parse_example(..., features=make_parse_example_spec(columns))
  prediction = linear_model(features, columns)
  ```

  The `sparse_combiner` argument works as follows
  For example, for two features represented as the categorical columns:

  ```python
    # Feature 1

    shape = [2, 2]
    {
        [0, 0]: "a"
        [0, 1]: "b"
        [1, 0]: "c"
    }

    # Feature 2

    shape = [2, 3]
    {
        [0, 0]: "d"
        [1, 0]: "e"
        [1, 1]: "f"
        [1, 2]: "f"
    }
  ```

  with `sparse_combiner` as "mean", the linear model outputs consequently
  are:

  ```
    y_0 = 1.0 / 2.0 * ( w_a + w_b ) + w_d + b
    y_1 = w_c + 1.0 / 3.0 * ( w_e + 2.0 * w_f ) + b
  ```

  where `y_i` is the output, `b` is the bias, and `w_x` is the weight
  assigned to the presence of `x` in the input features.

  Args:
    features: A mapping from key to tensors. `_FeatureColumn`s look up via these
      keys. For example `numeric_column('price')` will look at 'price' key in
      this dict. Values are `Tensor` or `SparseTensor` depending on
      corresponding `_FeatureColumn`.
    feature_columns: An iterable containing the FeatureColumns to use as inputs
      to your model. All items should be instances of classes derived from
      `_FeatureColumn`s.
    units: An integer, dimensionality of the output space. Default value is 1.
    sparse_combiner: A string specifying how to reduce if a categorical column
      is multivalent. Except `numeric_column`, almost all columns passed to
      `linear_model` are considered as categorical columns.  It combines each
      categorical column independently. Currently "mean", "sqrtn" and "sum" are
      supported, with "sum" the default for linear model. "sqrtn" often achieves
      good accuracy, in particular with bag-of-words columns.
        * "sum": do not
        normalize features in the column
        * "mean": do l1 normalization on features
        in the column
        * "sqrtn": do l2 normalization on features in the column
    weight_collections: A list of collection names to which the Variable will be
      added. Note that, variables will also be added to collections
      `tf.GraphKeys.GLOBAL_VARIABLES` and `ops.GraphKeys.MODEL_VARIABLES`.
    trainable: If `True` also add the variable to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    cols_to_vars: If not `None`, must be a dictionary that will be filled with a
      mapping from `_FeatureColumn` to associated list of `Variable`s.  For
      example, after the call, we might have cols_to_vars = { _NumericColumn(
      key='numeric_feature1', shape=(1,): [<tf.Variable
      'linear_model/price2/weights:0' shape=(1, 1)>], 'bias': [<tf.Variable
      'linear_model/bias_weights:0' shape=(1,)>], _NumericColumn(
      key='numeric_feature2', shape=(2,)): [<tf.Variable
      'linear_model/price1/weights:0' shape=(2, 1)>]} If a column creates no
      variables, its value will be an empty list. Note that cols_to_vars will
      also contain a string key 'bias' that maps to a list of Variables.

  Returns:
    A `Tensor` which represents predictions/logits of a linear model. Its shape
    is (batch_size, units) and its dtype is `float32`.

  Raises:
    ValueError: if an item in `feature_columns` is neither a `_DenseColumn`
      nor `_CategoricalColumn`.
  """
    with variable_scope.variable_scope(None, 'linear_model') as vs:
        model_name = _strip_leading_slashes(vs.name)
    linear_model_layer = _LinearModel(feature_columns=feature_columns, units=units, sparse_combiner=sparse_combiner, weight_collections=weight_collections, trainable=trainable, name=model_name)
    retval = linear_model_layer(features)
    if cols_to_vars is not None:
        cols_to_vars.update(linear_model_layer.cols_to_vars())
    return retval