import copy
import weakref
from tensorflow.python.checkpoint import save_util_v1
from tensorflow.python.checkpoint import trackable_view
from tensorflow.python.trackable import base
from tensorflow.python.util.tf_export import tf_export
def serialize_object_graph(self, saveables_cache=None):
    """Determine checkpoint keys for variables and build a serialized graph.

    Non-slot variables are keyed based on a shortest path from the root saveable
    to the object which owns the variable (i.e. the one which called
    `Trackable._add_variable` to create it).

    Slot variables are keyed based on a shortest path to the variable being
    slotted for, a shortest path to their optimizer, and the slot name.

    Args:
      saveables_cache: An optional cache storing previously created
        SaveableObjects created for each Trackable. Maps Trackables to a
        dictionary of attribute names to Trackable.

    Returns:
      A tuple of (named_variables, object_graph_proto, feed_additions):
        named_variables: A dictionary mapping names to variable objects.
        object_graph_proto: A TrackableObjectGraph protocol buffer
          containing the serialized object graph and variable references.
        feed_additions: A dictionary mapping from Tensors to values which should
          be fed when saving.

    Raises:
      ValueError: If there are invalid characters in an optimizer's slot names.
    """
    named_saveable_objects, object_graph_proto, feed_additions, _ = save_util_v1.serialize_object_graph_with_registered_savers(self, saveables_cache)
    return (named_saveable_objects, object_graph_proto, feed_additions)