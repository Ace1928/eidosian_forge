from tensorflow.python.util.tf_export import tf_export
from tensorflow.python import pywrap_tfe
@DeviceSpecV2.device_type.setter
def device_type(self, device_type):
    self._device_type = _as_device_str_or_none(device_type)
    self._as_string, self._hash = (None, None)