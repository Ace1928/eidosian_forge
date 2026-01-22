import abc
from tensorflow.python.keras.saving.saved_model import json_utils
from tensorflow.python.keras.saving.saved_model import utils
def trackable_children(self, serialization_cache):
    """Lists all Trackable children connected to this object."""
    if not utils.should_save_traces():
        return {}
    children = self.objects_to_serialize(serialization_cache)
    children.update(self.functions_to_serialize(serialization_cache))
    return children