import uuid
import tensorflow.compat.v2 as tf
from tensorflow.python.eager.context import get_device_name
def get_context_device_type():
    """Parse the current context and return the device type, eg CPU/GPU."""
    current_device = get_device_name()
    if current_device is None:
        return None
    return tf.compat.v1.DeviceSpec.from_string(current_device).device_type