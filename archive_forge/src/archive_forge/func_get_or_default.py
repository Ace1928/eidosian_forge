import abc
import os
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.checkpoint import checkpoint as trackable_util
from tensorflow.python.checkpoint import graph_view
from tensorflow.python.distribute import distribute_coordinator_context
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import resources
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.summary import summary
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import coordinator
from tensorflow.python.training import queue_runner
from tensorflow.python.training import saver as training_saver
from tensorflow.python.training import session_manager as sm
from tensorflow.python.training import session_run_hook
from tensorflow.python.util import function_utils
from tensorflow.python.util.tf_export import tf_export
@staticmethod
def get_or_default(arg_name, collection_key, default_constructor):
    """Get from cache or create a default operation."""
    elements = ops.get_collection(collection_key)
    if elements:
        if len(elements) > 1:
            raise RuntimeError('More than one item in the collection "%s". Please indicate which one to use by passing it to the tf.Scaffold constructor as:  tf.Scaffold(%s=item to use)', collection_key, arg_name)
        return elements[0]
    op = default_constructor()
    if op is not None:
        ops.add_to_collection(collection_key, op)
    return op