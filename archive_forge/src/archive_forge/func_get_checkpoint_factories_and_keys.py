import collections
from tensorflow.core.protobuf import trackable_object_graph_pb2
from tensorflow.python.checkpoint import saveable_compat
from tensorflow.python.checkpoint import util
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.saved_model import registration
from tensorflow.python.trackable import base
from tensorflow.python.trackable import python_state
from tensorflow.python.trackable import trackable_utils
from tensorflow.python.training.saving import saveable_object as saveable_object_lib
from tensorflow.python.training.saving import saveable_object_util
from tensorflow.python.util import object_identity
def get_checkpoint_factories_and_keys(object_names, object_map=None):
    """Gets a map of saveable factories and corresponding checkpoint keys.

  Args:
    object_names: a dictionary that maps `Trackable` objects to auto-generated
      string names.
    object_map: a dictionary mapping `Trackable` to copied `Trackable` objects.
      The copied objects are generated from `Trackable.
      _export_to_saved_model_graph()` which copies the object into another
      graph. Generally only resource objects (e.g. Variables, Tables) will be
      in this map.

  Returns:
    A tuple of (
      Dictionary mapping trackable -> list of _CheckpointFactoryData,
      Dictionary mapping registered saver name -> {object name -> trackable})
  """
    checkpoint_factory_map = object_identity.ObjectIdentityDictionary()
    unmapped_registered_savers = collections.defaultdict(dict)
    for trackable, object_name in object_names.items():
        object_to_save = util.get_mapped_trackable(trackable, object_map)
        saver_name = registration.get_registered_saver_name(object_to_save)
        if saver_name:
            unmapped_registered_savers[saver_name][object_name] = trackable
        else:
            checkpoint_factory_map[trackable] = []
            for name, saveable_factory in saveable_object_util.saveable_objects_from_trackable(object_to_save).items():
                key_suffix = saveable_compat.get_saveable_name(object_to_save) or name
                checkpoint_key = trackable_utils.checkpoint_key(object_name, key_suffix)
                if not saveable_compat.force_checkpoint_conversion_enabled():
                    name = key_suffix
                checkpoint_factory_map[trackable].append(_CheckpointFactoryData(factory=saveable_factory, name=name, checkpoint_key=checkpoint_key))
    return (checkpoint_factory_map, unmapped_registered_savers)