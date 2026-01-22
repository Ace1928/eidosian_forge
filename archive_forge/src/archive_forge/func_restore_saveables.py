from typing import Dict, List, Optional
import weakref
from tensorflow.core.protobuf import trackable_object_graph_pb2
from tensorflow.dtensor.python import api
from tensorflow.dtensor.python import d_variable
from tensorflow.dtensor.python import gen_dtensor_ops
from tensorflow.dtensor.python import layout
from tensorflow.dtensor.python import save_restore
from tensorflow.python.checkpoint import checkpoint as util
from tensorflow.python.checkpoint import checkpoint_options
from tensorflow.python.checkpoint import graph_view as graph_view_lib
from tensorflow.python.checkpoint import restore as restore_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.trackable import base
from tensorflow.python.trackable import data_structures
from tensorflow.python.training import py_checkpoint_reader
from tensorflow.python.training.saving import saveable_object
from tensorflow.python.training.saving import saveable_object_util
from tensorflow.python.util import deprecation
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
def restore_saveables(self, tensor_saveables: Dict[str, saveable_object.SaveableObject], python_positions: List[restore_lib.CheckpointPosition], registered_savers: Optional[Dict[str, Dict[str, base.Trackable]]]=None, reader: py_checkpoint_reader.NewCheckpointReader=None) -> Optional[List[ops.Operation]]:
    """Run or build restore operations for SaveableObjects.

    Args:
      tensor_saveables: `SaveableObject`s which correspond to Tensors.
      python_positions: `CheckpointPosition`s which correspond to `PythonState`
        Trackables bound to the checkpoint.
      registered_savers: a dict mapping saver names-> object name -> Trackable.
        This argument is not implemented for DTensorCheckpoint.
      reader: A CheckpointReader. Creates one lazily if None.

    Returns:
      When graph building, a list of restore operations, either cached or newly
      created, to restore `tensor_saveables`.
    """
    del registered_savers
    restore_ops = []
    if python_positions:
        if reader is None:
            reader = py_checkpoint_reader.NewCheckpointReader(self.save_path_string)
        for position in python_positions:
            key = position.object_proto.attributes[0].checkpoint_key
            position.trackable.deserialize(reader.get_tensor(key))
    if tensor_saveables:
        validated_saveables = saveable_object_util.validate_and_slice_inputs(tensor_saveables)
        validated_names = set((saveable.name for saveable in validated_saveables))
        if set(tensor_saveables.keys()) != validated_names:
            raise AssertionError('Saveable keys changed when validating. Got back %s, was expecting %s' % (tensor_saveables.keys(), validated_names))
        new_restore_ops = _DSaver(self._mesh, validated_saveables).restore(self.save_path_tensor, self.options)
        if not context.executing_eagerly():
            for name, restore_op in sorted(new_restore_ops.items()):
                restore_ops.append(restore_op)
                assert name not in self.restore_ops_by_name
                self.restore_ops_by_name[name] = restore_op
    return restore_ops