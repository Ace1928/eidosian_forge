import functools
from neutron_lib._i18n import _
import neutron_lib.exceptions as ne
from neutron_lib.tests import _base as base
def test_subnet_in_use(self):
    self._check_nexc(ne.SubnetInUse, _('Unable to complete operation on subnet garbage: not full.'), subnet_id='garbage', reason='not full')