import functools
from neutron_lib._i18n import _
import neutron_lib.exceptions as ne
from neutron_lib.tests import _base as base
def test_port_bound(self):
    self._check_nexc(ne.PortBound, _('Unable to complete operation on port bigmac, port is already bound, port type: ketchup, old_mac onions, new_mac salt.'), port_id='bigmac', vif_type='ketchup', old_mac='onions', new_mac='salt')