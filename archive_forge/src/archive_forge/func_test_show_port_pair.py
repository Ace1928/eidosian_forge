from unittest import mock
from osc_lib import exceptions
import testtools
from neutronclient.osc.v2.sfc import sfc_port_pair
from neutronclient.tests.unit.osc.v2.sfc import fakes
def test_show_port_pair(self):
    client = self.app.client_manager.network
    mock_port_pair_show = client.get_sfc_port_pair
    arglist = [self._port_pair_id]
    verifylist = [('port_pair', self._port_pair_id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    mock_port_pair_show.assert_called_once_with(self._port_pair_id)
    self.assertEqual(self.columns, columns)
    self.assertEqual(self.data, data)