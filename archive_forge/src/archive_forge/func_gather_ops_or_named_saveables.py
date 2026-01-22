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
def gather_ops_or_named_saveables(self):
    """Looks up or creates SaveableObjects which don't have cached ops.

    Returns:
      A tuple of (
          existing_restore_ops: list,
          named_saveables: dict,
          python_positions: list,
          registered_savers: dict)
    """
    recorded_registered_saver = self.get_registered_saver_name()
    if not (self.object_proto.attributes or recorded_registered_saver):
        return ([], {}, [], {})
    existing_restore_ops = []
    named_saveables = {}
    python_positions = []
    registered_savers = collections.defaultdict(dict)
    saveable_factories = saveable_object_util.saveable_objects_from_trackable(self.trackable)
    saver_name = registration.get_registered_saver_name(self.trackable)
    if recorded_registered_saver:
        if not self.skip_restore:
            name = self.object_proto.registered_saver.object_name
            registered_savers[recorded_registered_saver][name] = self.trackable
    elif saver_name:
        registered_savers[saver_name] = {self.object_proto.attributes[0].checkpoint_key: self.trackable}
    elif isinstance(self.trackable, python_state.PythonState):
        python_positions.append(self)
    elif saveable_factories.keys() == {trackable_utils.SERIALIZE_TO_TENSORS_NAME}:
        existing_restore_ops, named_saveables = self._create_serialize_to_tensor_saveable(saveable_factories)
    elif saveable_factories:
        existing_restore_ops, named_saveables = self._create_saveables_by_attribute_name(saveable_factories)
    else:
        for serialized_tensor in self.object_proto.attributes:
            self._checkpoint.unused_attributes.setdefault(self._proto_id, []).append(serialized_tensor.name)
    return (existing_restore_ops, named_saveables, python_positions, registered_savers)