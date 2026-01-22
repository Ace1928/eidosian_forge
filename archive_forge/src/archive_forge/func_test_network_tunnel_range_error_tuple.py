import functools
from neutron_lib._i18n import _
import neutron_lib.exceptions as ne
from neutron_lib.tests import _base as base
def test_network_tunnel_range_error_tuple(self):
    self._check_nexc(ne.NetworkTunnelRangeError, _("Invalid network tunnel range: '3:4' - present."), tunnel_range=(3, 4), error='present')