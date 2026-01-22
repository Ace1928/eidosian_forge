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
def generate_saveable_objects(checkpoint_factory_map, object_graph_proto=None, node_ids=None, object_map=None, call_with_mapped_captures=None, saveables_cache=None):
    """Create SaveableObjects and corresponding SerializedTensor protos."""
    named_saveable_objects = []
    if saveables_cache is None:
        feed_additions = None
    else:
        feed_additions = {}
    for trackable, factory_data_list in checkpoint_factory_map.items():
        fill_object_proto = object_graph_proto is not None and node_ids is not None
        if fill_object_proto:
            object_proto = object_graph_proto.nodes[node_ids[trackable]]
        object_to_save = util.get_mapped_trackable(trackable, object_map)
        if saveables_cache is not None:
            cached_attributes = saveables_cache.setdefault(object_to_save, {})
        else:
            cached_attributes = None
        for factory_data in factory_data_list:
            name = factory_data.name
            key = factory_data.checkpoint_key
            saveable_factory = factory_data.factory
            saveables = cached_attributes.get(name) if cached_attributes else None
            if saveables is not None:
                for saveable in saveables:
                    if key not in saveable.name:
                        saveables = None
                        del cached_attributes[name]
                        break
            if saveables is None:
                if callable(saveable_factory):
                    maybe_saveable = saveable_object_util.create_saveable_object(name, key, saveable_factory, call_with_mapped_captures)
                else:
                    maybe_saveable = saveable_factory
                if isinstance(maybe_saveable, saveable_object_lib.SaveableObject):
                    saveables = (maybe_saveable,)
                else:
                    saveables = tuple(saveable_object_util.saveable_objects_for_op(op=maybe_saveable, name=key))
                for saveable in saveables:
                    if key not in saveable.name:
                        raise AssertionError(f"The object {trackable} produced a SaveableObject with name '{saveable.name}' for attribute '{name}'. Expected a name containing '{key}'.")
                if cached_attributes is not None:
                    cached_attributes[name] = saveables
            if isinstance(object_to_save, python_state.PythonState):
                assert len(saveables) == 1
                saveable = saveables[0]
                if feed_additions is None:
                    assert saveables_cache is None
                    saveables = (saveable.freeze(),)
                else:
                    feed_additions.update(saveable.feed_dict_additions())
            named_saveable_objects.extend(saveables)
            if not fill_object_proto:
                continue
            if isinstance(saveables[0], saveable_object_util.TrackableSaveable) and (saveable_compat.force_checkpoint_conversion_enabled() or saveable_compat.get_saveable_name(object_to_save) is None):
                for local_name, local_key in saveables[0].get_proto_names_and_checkpoint_keys():
                    object_proto.attributes.add(name=local_name, checkpoint_key=local_key, full_name=util.get_full_name(object_to_save))
            else:
                object_proto.attributes.add(name=name, checkpoint_key=key, full_name=util.get_full_name(object_to_save))
    return (named_saveable_objects, feed_additions)