import collections
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.tpu.ops import tpu_ops
def hook_dummy_table_variables_to_activations(tpu_embedding, activations, dummy_table_variables):
    """Have activations depend on dummy table variables for gradient intercept.

  Args:
    tpu_embedding: TPUEmbedding, activations and dummy_table_variables are from
      tpu_embedding.
    activations: An OrderedDict of feature name String to activation tensors.
    dummy_table_variables: An OrderedDict of table name String to dummy table
      variables.

  Returns:
    An OrderedDict of feature name String to activation tensors, which can be
      used just as the activations input.
  """
    new_activations = collections.OrderedDict()
    for feature in activations:
        table = tpu_embedding.feature_to_config_dict[feature].table_id
        new_activations[feature] = tpu_ops.tpu_embedding_activations(dummy_table_variables[table], activations[feature], table_id=list(tpu_embedding.table_to_config_dict).index(table), lookup_id=tpu_embedding.table_to_features_dict[table].index(feature))
    return new_activations