from tensorflow.core.protobuf import trackable_object_graph_pb2
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.trackable import trackable_utils
from tensorflow.python.util import object_identity
def get_mapped_trackable(trackable, object_map):
    """Returns the mapped trackable if possible, otherwise returns trackable."""
    if object_map is None:
        return trackable
    else:
        return object_map.get(trackable, trackable)