from typing import List
from tensorflow.core.framework import full_type_pb2
from tensorflow.core.framework import types_pb2
from tensorflow.python.framework import type_spec
from tensorflow.python.ops.ragged.ragged_tensor import RaggedTensorSpec
from tensorflow.python.ops.structured.structured_tensor import StructuredTensor
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest
def iterator_full_type_from_spec(element_spec):
    """Returns a FullTypeDef for an iterator for the elements.

  Args:
     element_spec: A nested structure of `tf.TypeSpec` objects representing the
       element type specification.

  Returns:
    A FullTypeDef for an iterator for the element tensor representation.
  """
    args = fulltypes_for_flat_tensors(element_spec)
    return full_type_pb2.FullTypeDef(type_id=full_type_pb2.TFT_PRODUCT, args=[full_type_pb2.FullTypeDef(type_id=full_type_pb2.TFT_ITERATOR, args=[full_type_pb2.FullTypeDef(type_id=full_type_pb2.TFT_PRODUCT, args=args)])])