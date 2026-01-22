import functools
from neutron_lib._i18n import _
import neutron_lib.exceptions as ne
from neutron_lib.tests import _base as base
def test_ip_address_generation_failure(self):
    self._check_nexc(ne.IpAddressGenerationFailure, _('No more IP addresses available on network nuke.'), net_id='nuke')