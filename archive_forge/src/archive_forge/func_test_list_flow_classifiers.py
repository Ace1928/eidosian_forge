from unittest import mock
from osc_lib import exceptions
import testtools
from neutronclient.osc.v2.sfc import sfc_flow_classifier
from neutronclient.tests.unit.osc.v2.sfc import fakes
def test_list_flow_classifiers(self):
    arglist = []
    verifylist = []
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns = self.cmd.take_action(parsed_args)
    fcs = self.network.sfc_flow_classifiers()
    fc = fcs[0]
    data = [fc['id'], fc['name'], fc['protocol'], fc['source_ip_prefix'], fc['destination_ip_prefix'], fc['logical_source_port'], fc['logical_destination_port']]
    self.assertEqual(list(self.columns), columns[0])
    self.assertEqual(self.data, data)