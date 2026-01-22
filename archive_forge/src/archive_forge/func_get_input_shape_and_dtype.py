import numpy as np
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.util import nest
def get_input_shape_and_dtype(layer):
    """Retrieves input shape and input dtype of layer if applicable.

  Args:
    layer: Layer (or model) instance.

  Returns:
    Tuple (input_shape, input_dtype). Both could be None if the layer
      does not have a defined input shape.

  Raises:
    ValueError: in case an empty Sequential or Functional model is passed.
  """

    def _is_graph_model(layer):
        return hasattr(layer, '_is_graph_network') and layer._is_graph_network or layer.__class__.__name__ == 'Sequential'
    while _is_graph_model(layer):
        if not layer.layers:
            raise ValueError('An empty Model cannot be used as a Layer.')
        layer = layer.layers[0]
    if getattr(layer, '_batch_input_shape', None):
        return (layer._batch_input_shape, layer.dtype)
    return (None, None)