from tensorflow.python.eager import def_function
from tensorflow.python.keras.saving.saved_model import constants
from tensorflow.python.keras.saving.saved_model import save_impl
from tensorflow.python.keras.utils.generic_utils import LazyLoader
from tensorflow.python.trackable import base as trackable
from tensorflow.python.trackable.autotrackable import AutoTrackable
def set_and_validate_objects(self, object_dict):
    """Saves objects to a dictionary, and validates the values."""
    for key in self.all_checkpointable_objects:
        if key in object_dict:
            if not isinstance(object_dict[key], trackable.Trackable):
                raise ValueError('Object dictionary contained a non-trackable object: {} (for key {})'.format(object_dict[key], key))
            self._object_dict[key] = object_dict[key]
            setattr(self._keras_trackable, key, object_dict[key])
        else:
            raise ValueError('Object {} missing from serialized object dict.'.format(key))
    return self.checkpointable_objects