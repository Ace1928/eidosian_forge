import numpy as _np  # Avoids becoming a part of public Tensorflow API.
from tensorflow.compiler.tf2xla.python import xla as tf2xla
from tensorflow.compiler.xla import xla_data_pb2
from tensorflow.core.framework import attr_value_pb2
def mesh_split_sharding(device_mesh, tensor_split_dims_mapping, manual_mesh_dims=None):
    """Returns a Sharding object representing sharding along multiple dimensions.

  Args:
    device_mesh: An np.ndarray describing the topology of the device mesh and
      each element is the ID of the device in the topology.
    tensor_split_dims_mapping: A list of integers that map each tensor axis to
      the device mesh axis along which it is sharded. Its length is the tensor
      rank, and tensor_split_dims_mapping[i] is device mesh axis for tensor
      dimension i. Use -1 for tensor dimensions that are not sharded.
    manual_mesh_dims: An optional list of mesh dims for manual subgroups.

  Raises:
    ValueError: The number of tensor split dimensions is larger than device mesh
      rank.
  """
    manual_mesh_dims = manual_mesh_dims or []
    permutation = [d for d in tensor_split_dims_mapping if d >= 0] + manual_mesh_dims
    if len(permutation) > len(device_mesh.shape):
        raise ValueError('Number of tensor split dimensions (%r) is larger than device mesh rank (%r). tensor_split_dims_mapping: %r, device_mesh.shape: %r' % (len(permutation), len(device_mesh.shape), tensor_split_dims_mapping, device_mesh.shape))
    transpose_permutation = permutation + [d for d in range(len(device_mesh.shape)) if d not in permutation]
    tile_assignment = _np.transpose(device_mesh, transpose_permutation)
    tile_shape = [1 if d < 0 else device_mesh.shape[d] for d in tensor_split_dims_mapping + manual_mesh_dims]
    subgroup_modes = [xla_data_pb2.OpSharding.MANUAL] * len(manual_mesh_dims)
    partial = len(permutation) < len(device_mesh.shape)
    if partial:
        tile_shape.append(_np.prod(device_mesh.shape) // _np.prod(tile_shape))
        subgroup_modes.append(xla_data_pb2.OpSharding.REPLICATED)
    tile_assignment = _np.reshape(tile_assignment, tile_shape)
    if manual_mesh_dims:
        return Sharding.subgroup_tile(tile_assignment, subgroup_modes)
    if partial:
        return Sharding.partial_tile(tile_assignment)
    return Sharding.tile(tile_assignment)