import operator
from tensorflow.core.framework import versions_pb2
from tensorflow.core.protobuf import saved_object_graph_pb2
from tensorflow.python.trackable import data_structures
from tensorflow.python.util.tf_export import tf_export
def should_load(self, proto):
    """Checks if this object should load the SavedUserObject `proto`."""
    if proto.identifier != self.identifier:
        return False
    if self.version < proto.version.min_consumer:
        return False
    if proto.version.producer < self._min_producer_version:
        return False
    for bad_version in proto.version.bad_consumers:
        if self.version == bad_version:
            return False
    return True