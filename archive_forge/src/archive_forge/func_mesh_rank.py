import numpy as np
from tensorflow.core.protobuf.tpu import topology_pb2
from tensorflow.python.util.tf_export import tf_export
@property
def mesh_rank(self):
    """Returns the number of dimensions in the mesh."""
    return len(self._mesh_shape)