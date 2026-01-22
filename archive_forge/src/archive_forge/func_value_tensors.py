import collections
from tensorflow.python.checkpoint import checkpoint_view
from tensorflow.python.checkpoint import functional_saver
from tensorflow.python.checkpoint import save_util_v1
from tensorflow.python.checkpoint import saveable_compat
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_io_ops as io_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import registration
from tensorflow.python.trackable import base
from tensorflow.python.trackable import constants
from tensorflow.python.trackable import python_state
from tensorflow.python.trackable import trackable_utils
from tensorflow.python.training.saving import saveable_object_util
from tensorflow.python.util import object_identity
def value_tensors(self, shape_and_slices=None):
    """Create value `Tensor`s for this object's attributes.

    Does not require that the Python object has been created. Used for
    restore-on-create when executing eagerly.

    Args:
      shape_and_slices: A dict mapping from object attribute names to a shape
        and slice string that will be passed to a RestoreV2 op. If the dict is
        None or if an object attribute is not in the dict, the full tensor will
        be restored.

    Returns:
      A dictionary mapping from object attribute names to `Tensor`s.
    """
    value_tensors = {}
    for serialized_tensor in self.object_proto.attributes:
        checkpoint_key = serialized_tensor.checkpoint_key
        dtype = self._checkpoint.dtype_map[checkpoint_key]
        base_type = dtype.base_dtype
        io_device = self._checkpoint.options.experimental_io_device or 'cpu:0'
        with ops.init_scope():
            with ops.device(io_device):
                if shape_and_slices is not None and serialized_tensor.name in shape_and_slices:
                    shape_and_slice = shape_and_slices[serialized_tensor.name]
                else:
                    shape_and_slice = ''
                value, = io_ops.restore_v2(prefix=self._checkpoint.save_path_tensor, tensor_names=[checkpoint_key], shape_and_slices=[shape_and_slice], dtypes=[base_type], name='%s_checkpoint_read' % (serialized_tensor.name,))
            value_tensors[serialized_tensor.name] = array_ops.identity(value)
    return value_tensors