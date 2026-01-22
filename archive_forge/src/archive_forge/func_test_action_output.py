import logging
from os_ken.tests.integrated import tester
from os_ken.ofproto import ofproto_v1_0
from os_ken.ofproto import ether
from os_ken.ofproto import nx_match
def test_action_output(self, dp):
    out_port = 2
    self._verify = [dp.ofproto.OFPAT_OUTPUT, 'port', out_port]
    action = dp.ofproto_parser.OFPActionOutput(out_port)
    self.add_action(dp, [action])