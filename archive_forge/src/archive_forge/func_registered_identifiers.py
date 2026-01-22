import operator
from tensorflow.core.framework import versions_pb2
from tensorflow.core.protobuf import saved_object_graph_pb2
from tensorflow.python.trackable import data_structures
from tensorflow.python.util.tf_export import tf_export
@tf_export('__internal__.saved_model.load.registered_identifiers', v1=[])
def registered_identifiers():
    """Return all the current registered revived object identifiers.

  Returns:
    A set of strings.
  """
    return _REVIVED_TYPE_REGISTRY.keys()