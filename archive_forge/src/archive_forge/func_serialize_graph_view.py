import collections
from typing import Any, Callable, List, Optional, Tuple, Mapping, Union, Dict
from tensorflow.core.protobuf import trackable_object_graph_pb2
from tensorflow.python.checkpoint import graph_view as graph_view_lib
from tensorflow.python.checkpoint import save_util_v1
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
from tensorflow.python.types import core
from tensorflow.python.util import object_identity
def serialize_graph_view(graph_view: graph_view_lib.ObjectGraphView, object_map: Optional[Mapping[base.Trackable, base.Trackable]]=None, call_with_mapped_captures: Optional[Callable[..., Any]]=None, cache: Optional[Dict[base.Trackable, Any]]=None) -> ...:
    """Gathers serialization objects, and creates a TrackableObjectGraph proto."""
    trackable_data, node_ids = _gather_trackable_data(graph_view, object_map)
    tensor_trackables, pystate_trackables, registered_trackables = _split_trackables(trackable_data)
    object_graph_proto = _fill_object_graph_proto(trackable_data)
    serialized_tensors = _get_and_write_tensors_to_serialize(tensor_trackables, node_ids, call_with_mapped_captures, cache, object_graph_proto)
    registered_savers = _get_and_write_registered_savers(registered_trackables, object_graph_proto)
    if cache is None:
        feed_additions = None
        serialized_tensors.update(_get_and_write_tensors_to_serialize(pystate_trackables, node_ids, call_with_mapped_captures, cache, object_graph_proto))
    else:
        new_serialized_tensors, feed_additions = _get_and_write_pystate_feed_additions(pystate_trackables, cache, object_graph_proto)
        serialized_tensors.update(new_serialized_tensors)
    util.add_checkpoint_values_check(object_graph_proto)
    return (serialized_tensors, feed_additions, registered_savers, object_graph_proto)