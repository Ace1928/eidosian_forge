import operator
from unittest import mock
from osc_lib.tests.utils import ParserException
from osc_lib import utils as osc_utils
from osc_lib.utils import columns as column_util
from neutronclient.tests.unit.osc.v2.networking_bgpvpn import fakes
def test_router_association_unknown_arg(self):
    arglist = self._build_args('--unknown arg')
    try:
        self._exec_create_router_association(None, arglist, None)
    except ParserException as e:
        self.assertEqual(format(e), 'Argument parse failed')