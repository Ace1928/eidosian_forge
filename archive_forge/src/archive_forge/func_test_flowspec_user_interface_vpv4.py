import logging
import os
import sys
import unittest
from os_ken.utils import binary_str
from os_ken.lib import pcaplib
from os_ken.lib.packet import packet
from os_ken.lib.packet import bgp
from os_ken.lib.packet import afi
from os_ken.lib.packet import safi
def test_flowspec_user_interface_vpv4(self):
    rules = RULES_BASE + [bgp.FlowSpecDestPrefix(addr='10.0.0.0', length=24), bgp.FlowSpecSrcPrefix(addr='20.0.0.0', length=24), bgp.FlowSpecIPProtocol(operator=bgp.FlowSpecIPProtocol.EQ, value=6), bgp.FlowSpecFragment(operator=0, value=bgp.FlowSpecFragment.LF), bgp.FlowSpecFragment(operator=bgp.FlowSpecFragment.MATCH, value=bgp.FlowSpecFragment.FF), bgp.FlowSpecFragment(operator=bgp.FlowSpecFragment.AND | bgp.FlowSpecFragment.MATCH, value=bgp.FlowSpecFragment.ISF), bgp.FlowSpecFragment(operator=bgp.FlowSpecFragment.NOT, value=bgp.FlowSpecFragment.DF)]
    msg = bgp.FlowSpecVPNv4NLRI.from_user(route_dist='65001:250', dst_prefix='10.0.0.0/24', src_prefix='20.0.0.0/24', ip_proto='6', port='>=8000 & <=9000 | ==80', dst_port='8080 >9000&<9050 | <=1000', src_port='<=9090 & >=9080 <10100 & >10000', icmp_type=0, icmp_code=6, tcp_flags='SYN+ACK & !=URGENT', packet_len='1000 & 1100', dscp='22 24', fragment='LF ==FF&==ISF | !=DF')
    msg2 = bgp.FlowSpecVPNv4NLRI(route_dist='65001:250', rules=rules)
    binmsg = msg.serialize()
    binmsg2 = msg2.serialize()
    self.assertEqual(str(msg), str(msg2))
    self.assertEqual(binary_str(binmsg), binary_str(binmsg2))
    msg3, rest = bgp.FlowSpecVPNv4NLRI.parser(binmsg)
    self.assertEqual(str(msg), str(msg3))
    self.assertEqual(rest, b'')