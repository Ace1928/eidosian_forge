import collections
import re
def compute_capability_from_device_desc(device_attrs):
    """Returns the GpuInfo given a DeviceAttributes proto.

  Args:
    device_attrs: A DeviceAttributes proto.

  Returns
    A gpu_info tuple. Both fields are None if `device_attrs` does not have a
    valid physical_device_desc field.
  """
    match = _PHYSICAL_DEVICE_DESCRIPTION_REGEX.search(device_attrs.physical_device_desc)
    if not match:
        return GpuInfo(None, None)
    cc = (int(match.group(2)), int(match.group(3))) if match.group(2) else None
    return GpuInfo(match.group(1), cc)