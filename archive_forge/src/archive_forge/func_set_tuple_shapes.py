import itertools
import numpy as np
from tensorflow.python.compiler.xla.experimental import xla_sharding
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.tpu import tpu_name_util
from tensorflow.python.tpu import tpu_sharding
from tensorflow.python.tpu.ops import tpu_ops
from tensorflow.python.util import nest
def set_tuple_shapes(self, tuple_shapes):
    """Sets the shape of each element of the queue.

    tuple_shapes must be a list of length
    self.number_of_tuple_elements, and each element must be
    convertible to a TensorShape.

    Args:
      tuple_shapes: the shapes of each queue element.

    Raises:
      ValueError: if tuple_shapes is not of length
        self.number_of_tuple_elements.
      TypeError: if an element of tuple_shapes cannot be converted to
        a TensorShape.
    """
    if len(tuple_shapes) != self.number_of_tuple_elements:
        raise ValueError(f'tuple_shapes is {str(tuple_shapes)}, but must be a list of length {self.number_of_tuple_elements}')
    try:
        tuple_shapes = [tensor_shape.as_shape(shape) for shape in tuple_shapes]
    except (ValueError, TypeError) as e:
        raise TypeError(f'tuple_shapes is {str(tuple_shapes)}, but must be a list of elements each convertible to TensorShape: got error {str(e)}') from e
    if self._frozen:
        for frozen, updated in zip(self._tuple_shapes, tuple_shapes):
            if frozen != updated:
                raise ValueError(f'Trying to update InfeedQueue with frozen configuration with an incompatible shape. Frozen shapes are {str(self._tuple_shapes)}, updated shapes are {str(tuple_shapes)}')
    else:
        self._tuple_shapes = tuple_shapes
    self._validate()