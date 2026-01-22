import collections
import re
from keras.src.api_export import keras_export
from keras.src.backend.common import global_state
def get_object_name(obj):
    if hasattr(obj, 'name'):
        return obj.name
    elif hasattr(obj, '__name__'):
        return to_snake_case(obj.__name__)
    elif hasattr(obj, '__class__'):
        return to_snake_case(obj.__class__.__name__)
    return to_snake_case(str(obj))