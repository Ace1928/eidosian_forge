import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import cond as tf_cond
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_linalg_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import map_fn
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import special_math_ops
from tensorflow.python.ops import stateless_random_ops
from tensorflow.python.ops import while_loop
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
def orthogonalize_close_eigenvectors(eigenvectors):

    def orthogonalize_cluster(cluster_idx, eigenvectors):
        start = ortho_interval_start[cluster_idx]
        end = ortho_interval_end[cluster_idx]
        update_indices = array_ops.expand_dims(math_ops.range(start, end), -1)
        vectors_in_cluster = eigenvectors[start:end, :]
        q, _ = qr(transpose(vectors_in_cluster))
        vectors_to_update = transpose(q)
        eigenvectors = array_ops.tensor_scatter_nd_update(eigenvectors, update_indices, vectors_to_update)
        return (cluster_idx + 1, eigenvectors)
    _, eigenvectors = while_loop.while_loop(lambda i, ev: math_ops.less(i, num_clusters), orthogonalize_cluster, [0, eigenvectors])
    return eigenvectors