from unittest import mock
from osc_lib import exceptions
import testtools
from neutronclient.osc.v2.sfc import sfc_port_pair
from neutronclient.tests.unit.osc.v2.sfc import fakes
def test_create_port_pair_default_options(self):
    arglist = ['--ingress', self._port_pair['ingress'], '--egress', self._port_pair['egress'], self._port_pair['name']]
    verifylist = [('ingress', self._port_pair['ingress']), ('egress', self._port_pair['egress']), ('name', self._port_pair['name'])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.network.create_sfc_port_pair.assert_called_once_with(**{'name': self._port_pair['name'], 'ingress': self._port_pair['ingress'], 'egress': self._port_pair['egress']})
    self.assertEqual(self.columns, columns)
    self.assertEqual(self.data, data)