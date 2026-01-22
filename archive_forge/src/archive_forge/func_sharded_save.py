import collections
from typing import Dict, List, Union
from tensorflow.dtensor.python import api
from tensorflow.dtensor.python import d_variable
from tensorflow.dtensor.python import gen_dtensor_ops
from tensorflow.dtensor.python import layout as layout_lib
from tensorflow.dtensor.python import mesh_util
from tensorflow.python.eager import context
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.ops import io_ops
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.util.tf_export import tf_export
@tf_export('experimental.dtensor.sharded_save', v1=[])
def sharded_save(mesh: layout_lib.Mesh, file_prefix: Union[str, tensor_lib.Tensor], tensor_names: Union[List[str], tensor_lib.Tensor], shape_and_slices: Union[List[str], tensor_lib.Tensor], tensors: List[Union[tensor_lib.Tensor, tf_variables.Variable]]):
    """Saves given named tensor slices in a sharded, multi-client safe fashion.

  The method makes sure the checkpoint directory state is correct in a sharded
  mutli-client saving. Namely, we place a barrier after SaveV2 to make sure
  every client has done writing the files. And another one after
  MergeV2Checkpoints to make sure all Metadata is properly merged.

  Upon existing, the checkpoint is completed and the all directory operations
  are done.

  Args:
    mesh: The Mesh that contains the Tensors to save.
    file_prefix: The prefix of checkpoint.
    tensor_names: a list of tensor names used in save op.
    shape_and_slices: a list of shape and slice specification used in save op.
      The only supported value is "" as we don't support distributed saving with
      slices yet.
    tensors: a list of tensors used in save op. The order should match
      tensor_names.

  Returns:
    A MergeV2Checkpoints op that merged all Metadata.
  """
    with ops.device(api.device_name()):
        io_ops.save_v2(file_prefix, tensor_names, shape_and_slices, tensors)
    mesh_util.barrier(mesh.host_mesh(), 'SaveV2')
    with api.default_mesh(mesh.host_mesh()):
        merge_op = io_ops.MergeV2Checkpoints(checkpoint_prefixes=[file_prefix], destination_prefix=file_prefix, delete_old_dirs=True)
    mesh_util.barrier(mesh.host_mesh(), 'MergeV2Checkpoints')
    return merge_op