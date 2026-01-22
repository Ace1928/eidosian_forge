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
def serialize_gathered_objects(graph_view, object_map=None, call_with_mapped_captures=None, saveables_cache=None):
    """Create SaveableObjects and protos for gathered objects."""
    trackable_objects, node_paths = graph_view.breadth_first_traversal()
    object_names = object_identity.ObjectIdentityDictionary()
    for obj, path in node_paths.items():
        object_names[obj] = trackable_utils.object_path_to_string(path)
    node_ids = object_identity.ObjectIdentityDictionary()
    for node_id, node in enumerate(trackable_objects):
        node_ids[node] = node_id
    slot_variables = util.serialize_slot_variables(trackable_objects=trackable_objects, node_ids=node_ids, object_names=object_names)
    object_graph_proto = _fill_object_graph_proto(graph_view=graph_view, trackable_objects=trackable_objects, node_ids=node_ids, slot_variables=slot_variables)
    named_saveable_objects, feed_additions, registered_savers = _add_attributes_to_object_graph(trackable_objects=trackable_objects, object_graph_proto=object_graph_proto, node_ids=node_ids, object_names=object_names, object_map=object_map, call_with_mapped_captures=call_with_mapped_captures, saveables_cache=saveables_cache)
    util.add_checkpoint_values_check(object_graph_proto)
    return (named_saveable_objects, object_graph_proto, feed_additions, registered_savers)