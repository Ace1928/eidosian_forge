import functools
from neutron_lib._i18n import _
import neutron_lib.exceptions as ne
from neutron_lib.tests import _base as base
def test_bad_request_misused(self):
    try:
        self._check_nexc(ne.BadRequest, _('Bad A request: B.'), resource='A', msg='B')
    except AttributeError:
        pass