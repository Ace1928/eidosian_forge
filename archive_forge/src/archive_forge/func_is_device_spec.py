from tensorflow.python import tf2
from tensorflow.python.framework import device_spec
def is_device_spec(obj):
    """Abstract away the fact that DeviceSpecV2 is the base class."""
    return isinstance(obj, device_spec.DeviceSpecV2)