from tensorflow.python import tf2
from tensorflow.python.framework import device_spec
def canonical_name(device):
    """Returns a canonical name for the given `DeviceSpec` or device name."""
    if device is None:
        return ''
    if is_device_spec(device):
        return device.to_string()
    else:
        device = DeviceSpec.from_string(device)
        return device.to_string()