from tensorflow.python.ops import check_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.util import dispatch
@dispatch.dispatch_for_api(check_ops.assert_type)
def assert_type(tensor: ragged_tensor.Ragged, tf_type, message=None, name=None):
    return check_ops.assert_type(tensor.flat_values, tf_type, message=message, name=name)