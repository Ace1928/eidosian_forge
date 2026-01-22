import functools
from neutron_lib._i18n import _
import neutron_lib.exceptions as ne
from neutron_lib.tests import _base as base
def test_device_not_found_error(self):
    self._check_nexc(ne.DeviceNotFoundError, _("Device 'device' does not exist."), device_name='device')