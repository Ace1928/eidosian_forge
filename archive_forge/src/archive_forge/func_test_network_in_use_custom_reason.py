import functools
from neutron_lib._i18n import _
import neutron_lib.exceptions as ne
from neutron_lib.tests import _base as base
def test_network_in_use_custom_reason(self):
    self._check_nexc(ne.NetworkInUse, _('Unable to complete operation on network foo. not full.'), net_id='foo', reason='not full')