from unittest import mock
from osc_lib import exceptions
import testtools
from neutronclient.osc.v2.sfc import sfc_port_pair_group
from neutronclient.tests.unit.osc.v2.sfc import fakes
def test_create_port_pair_group_default_options(self):
    arglist = ['--port-pair', self._port_pair_group['port_pairs'], self._port_pair_group['name']]
    verifylist = [('port_pairs', [self._port_pair_group['port_pairs']]), ('name', self._port_pair_group['name'])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.network.create_sfc_port_pair_group.assert_called_once_with(**{'name': self._port_pair_group['name'], 'port_pairs': [self._port_pair_group['port_pairs']]})
    self.assertEqual(self.columns, columns)
    self.assertEqual(self.data, data)