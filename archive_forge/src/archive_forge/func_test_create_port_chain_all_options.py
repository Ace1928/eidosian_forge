from unittest import mock
from osc_lib import exceptions
import testtools
from neutronclient.osc.v2.sfc import sfc_port_chain
from neutronclient.tests.unit.osc.v2.sfc import fakes
def test_create_port_chain_all_options(self):
    arglist = ['--description', self._port_chain['description'], '--port-pair-group', self._port_chain['port_pair_groups'], self._port_chain['name'], '--flow-classifier', self._port_chain['flow_classifiers'], '--chain-parameters', 'correlation=mpls,symmetric=true']
    cp = {'correlation': 'mpls', 'symmetric': 'true'}
    verifylist = [('port_pair_groups', [self._port_chain['port_pair_groups']]), ('name', self._port_chain['name']), ('description', self._port_chain['description']), ('flow_classifiers', [self._port_chain['flow_classifiers']]), ('chain_parameters', [cp])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.network.create_sfc_port_chain.assert_called_once_with(**{'name': self._port_chain['name'], 'port_pair_groups': [self._port_chain['port_pair_groups']], 'description': self._port_chain['description'], 'flow_classifiers': [self._port_chain['flow_classifiers']], 'chain_parameters': cp})
    self.assertEqual(self.columns, columns)
    self.assertEqual(self.data, data)