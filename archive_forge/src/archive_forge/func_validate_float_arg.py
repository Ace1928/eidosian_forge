import math
from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.utils.numerical_utils import normalize
def validate_float_arg(value, name):
    """check penalty number availability, raise ValueError if failed."""
    if not isinstance(value, (float, int)) or (math.isinf(value) or math.isnan(value)) or value < 0:
        raise ValueError(f'Invalid value for argument {name}: expected a non-negative float.Received: {name}={value}')
    return float(value)