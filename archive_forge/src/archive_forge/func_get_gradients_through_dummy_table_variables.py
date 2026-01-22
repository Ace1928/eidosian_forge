import collections
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.tpu.ops import tpu_ops
def get_gradients_through_dummy_table_variables(tpu_embedding):
    """Get gradients wrt the activations of each feature.

  Args:
    tpu_embedding: TPUEmbedding, create dummy table variable to be used with
      tpu_embedding.

  Returns:
    An OrderedDict mapping feature name to gradient.

  Raises:
    ValueError: if some gradients are not defined.
  """
    g = ops.get_default_graph()
    gradients_found = False
    for table_id, table in enumerate(tpu_embedding.table_to_config_dict):
        table_gradients = g.get_collection('tpu_embedding_gradients_table_{}'.format(table_id))
        if any((gradient is None for gradient in table_gradients)):
            logging.warn('Table {} with id {} has undefined gradients: this is probably because the model asked TPUEmbedding to compute activations that were not used, or tf.stop_gradient() is applied. Gradients of zeros are sent back to TPUEmbedding instead. Gradients of zeros and no gradients are equivalent for SGD, AdaGrad, FTRL, etc, but might differ for other optimizers due to implementation of TPU embedding optimizers.'.format(table, table_id))
        gradients_found = gradients_found or any((gradient is not None for gradient in table_gradients))
    if not gradients_found:
        logging.warn('All tables have undefined gradients: this is probably because the model asked TPUEmbedding to compute activations that were not used. If all TPUEmbedding features have stop_gradients, consider using the INFERENCE mode instead.')
    feature_to_gradient_dict = collections.OrderedDict()
    for table_id, table in enumerate(tpu_embedding.table_to_config_dict):
        table_gradients = g.get_collection('tpu_embedding_gradients_table_{}'.format(table_id))
        for feature, gradient in zip(tpu_embedding.table_to_features_dict[table], table_gradients):
            if gradient is not None:
                feature_to_gradient_dict[feature] = gradient
            else:
                dimension = tpu_embedding.table_to_config_dict[table].dimension
                batch_size = tpu_embedding.batch_size_per_core
                max_sequence_length = tpu_embedding.feature_to_config_dict[feature].max_sequence_length
                if max_sequence_length:
                    feature_to_gradient_dict[feature] = array_ops.zeros([batch_size, max_sequence_length, dimension])
                else:
                    feature_to_gradient_dict[feature] = array_ops.zeros([batch_size, dimension])
    return feature_to_gradient_dict