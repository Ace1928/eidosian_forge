import collections
from tensorflow.core.protobuf import trackable_object_graph_pb2
from tensorflow.python.checkpoint import trackable_view
from tensorflow.python.framework import errors_impl
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.trackable import base
from tensorflow.python.training import py_checkpoint_reader
from tensorflow.python.util import object_identity
from tensorflow.python.util.tf_export import tf_export
def descendants(self):
    """Returns a list of trackables by node_id attached to obj."""
    return list(self._descendants_with_paths().keys())