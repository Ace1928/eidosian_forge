from unittest import mock
from osc_lib import exceptions
from osc_lib.tests import utils as tests_utils
import testtools
from neutronclient.osc.v2.sfc import sfc_service_graph
from neutronclient.tests.unit.osc.v2.sfc import fakes
def test_list_sfc_service_graphs(self):
    arglist = []
    verifylist = []
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns = self.cmd.take_action(parsed_args)[0]
    sgs = self.network.sfc_service_graphs()
    sg = sgs[0]
    data = [sg['id'], sg['name'], sg['port_chains']]
    self.assertEqual(list(self.columns), columns)
    self.assertEqual(self.data, data)