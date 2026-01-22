from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import tensor_array_ops
def is_dense_tensor(t):
    return tensor_util.is_tf_type(t) and (not isinstance(t, sparse_tensor.SparseTensor))