import functools
from neutron_lib._i18n import _
import neutron_lib.exceptions as ne
from neutron_lib.tests import _base as base
def test_mac_address_in_use(self):
    self._check_nexc(ne.MacAddressInUse, _('Unable to complete operation for network nutters. The mac address grill is in use.'), net_id='nutters', mac='grill')