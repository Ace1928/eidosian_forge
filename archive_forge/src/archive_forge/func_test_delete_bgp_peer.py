from unittest import mock
from neutronclient.osc.v2.dynamic_routing import bgp_peer
from neutronclient.tests.unit.osc.v2.dynamic_routing import fakes
def test_delete_bgp_peer(self):
    arglist = [self._bgp_peer['name']]
    verifylist = [('bgp_peer', self._bgp_peer['name'])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.networkclient.delete_bgp_peer.assert_called_once_with(self._bgp_peer['name'])
    self.assertIsNone(result)