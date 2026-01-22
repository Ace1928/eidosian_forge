import functools
from neutron_lib._i18n import _
import neutron_lib.exceptions as ne
from neutron_lib.tests import _base as base
def test_service_port_in_use(self):
    self._check_nexc(ne.ServicePortInUse, _('Port harbor cannot be deleted directly via the port API: docking.'), port_id='harbor', reason='docking')