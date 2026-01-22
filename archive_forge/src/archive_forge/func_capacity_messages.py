from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
@property
def capacity_messages(self):
    """A map of enum to user-friendly message."""
    if self._capacity_messages_cache is None:
        device_capacity_enum_android = self.context['testing_messages'].PerAndroidVersionInfo.DeviceCapacityValueValuesEnum
        device_capacity_enum_ios = self.context['testing_messages'].PerIosVersionInfo.DeviceCapacityValueValuesEnum
        self._capacity_messages_cache = {device_capacity_enum_android.DEVICE_CAPACITY_UNSPECIFIED: 'None', device_capacity_enum_android.DEVICE_CAPACITY_HIGH: 'High', device_capacity_enum_android.DEVICE_CAPACITY_MEDIUM: 'Medium', device_capacity_enum_android.DEVICE_CAPACITY_LOW: 'Low', device_capacity_enum_android.DEVICE_CAPACITY_NONE: 'None', device_capacity_enum_ios.DEVICE_CAPACITY_UNSPECIFIED: 'None', device_capacity_enum_ios.DEVICE_CAPACITY_HIGH: 'High', device_capacity_enum_ios.DEVICE_CAPACITY_MEDIUM: 'Medium', device_capacity_enum_ios.DEVICE_CAPACITY_LOW: 'Low', device_capacity_enum_ios.DEVICE_CAPACITY_NONE: 'None'}
    return self._capacity_messages_cache