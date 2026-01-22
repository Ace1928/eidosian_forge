import abc
from tensorflow.python.keras.saving.saved_model import json_utils
from tensorflow.python.keras.saving.saved_model import utils
@property
def tracking_metadata(self):
    """String stored in metadata field in the SavedModel proto.

    Returns:
      A serialized JSON storing information necessary for recreating this layer.
    """
    return json_utils.Encoder().encode(self.python_properties)