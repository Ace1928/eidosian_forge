from unittest import mock
from osc_lib import exceptions
import testtools
from neutronclient.osc.v2.sfc import sfc_port_pair
from neutronclient.tests.unit.osc.v2.sfc import fakes
def test_create_port_pair_all_options(self):
    self._test_create_port_pair_all_options('None')