from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import type_spec as type_spec_module
from tensorflow.python.keras.utils import object_identity
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.ragged import ragged_operators  # pylint: disable=unused-import
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.util import nest
def set_shape(self, shape):
    """Updates the shape of this KerasTensor. Mimics `tf.Tensor.set_shape()`."""
    if not isinstance(shape, tensor_shape.TensorShape):
        shape = tensor_shape.TensorShape(shape)
    if shape.dims is not None:
        dim_list = [dim.value for dim in shape.dims]
        for dim in range(len(dim_list)):
            if dim_list[dim] is None and self.shape.dims is not None:
                dim_list[dim] = self.shape.dims[dim]
        shape = tensor_shape.TensorShape(dim_list)
    if not self.shape.is_compatible_with(shape):
        raise ValueError("Keras symbolic input/output's shape %s is notcompatible with supplied shape %s" % (self.shape, shape))
    else:
        self._type_spec._shape = shape