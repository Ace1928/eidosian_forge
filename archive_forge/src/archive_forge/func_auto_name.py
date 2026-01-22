import collections
import re
from keras.src.api_export import keras_export
from keras.src.backend.common import global_state
def auto_name(prefix):
    prefix = to_snake_case(prefix)
    return uniquify(prefix)