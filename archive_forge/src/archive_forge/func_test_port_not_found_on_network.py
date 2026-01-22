import functools
from neutron_lib._i18n import _
import neutron_lib.exceptions as ne
from neutron_lib.tests import _base as base
def test_port_not_found_on_network(self):
    self._check_nexc(ne.PortNotFoundOnNetwork, _('Port serial could not be found on network USB.'), port_id='serial', net_id='USB')