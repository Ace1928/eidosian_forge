from unittest import mock
from neutronclient.osc.v2.dynamic_routing import bgp_peer
from neutronclient.tests.unit.osc.v2.dynamic_routing import fakes
def test_bgp_peer_show(self):
    arglist = [self._bgp_peer_name]
    verifylist = [('bgp_peer', self._bgp_peer_name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    data = self.cmd.take_action(parsed_args)
    self.networkclient.get_bgp_peer.assert_called_once_with(self._bgp_peer_name)
    self.assertEqual(self.columns, data[0])
    self.assertEqual(self.data, data[1])