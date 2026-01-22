import abc
from tensorflow.python.keras.saving.saved_model import json_utils
from tensorflow.python.keras.saving.saved_model import utils
@abc.abstractproperty
def python_properties(self):
    """Returns dictionary of python properties to save in the metadata.

    This dictionary must be serializable and deserializable to/from JSON.

    When loading, the items in this dict are used to initialize the object and
    define attributes in the revived object.
    """
    raise NotImplementedError