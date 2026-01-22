import functools
import operator
from typing import Optional, Sequence, Type
from tensorflow.core.framework import tensor_shape_pb2
from tensorflow.core.function import trace_type
from tensorflow.core.protobuf import struct_pb2
from tensorflow.python import tf2
from tensorflow.python.eager import monitoring
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.types import trace
from tensorflow.python.util.tf_export import tf_export
from tensorflow.tools.docs import doc_controls
def unknown_shape(rank=None, **kwargs):
    """Returns an unknown TensorShape, optionally with a known rank.

  Args:
    rank: (Optional) If specified, the number of dimensions in the shape.
    **kwargs: For backwards compatibility.

  Returns:
    An unknown TensorShape.

  Raises:
    TypeError: In case of invalid arguments.
  """
    if rank is None and 'ndims' in kwargs:
        rank = kwargs.pop('ndims')
    if kwargs:
        raise TypeError('Unknown argument: %s' % kwargs)
    if rank is None:
        return TensorShape(None)
    else:
        return TensorShape([Dimension(None)] * rank)