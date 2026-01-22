import functools
from neutron_lib._i18n import _
import neutron_lib.exceptions as ne
from neutron_lib.tests import _base as base
def test_preexisting_device_failure(self):
    self._check_nexc(ne.PreexistingDeviceFailure, _('Creation failed. hal9000 already exists.'), dev_name='hal9000')