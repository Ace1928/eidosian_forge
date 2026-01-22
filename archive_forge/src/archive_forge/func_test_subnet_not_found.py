import functools
from neutron_lib._i18n import _
import neutron_lib.exceptions as ne
from neutron_lib.tests import _base as base
def test_subnet_not_found(self):
    self._check_nexc(ne.SubnetNotFound, _('Subnet root could not be found.'), subnet_id='root')