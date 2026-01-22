import logging
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ether
from os_ken.ofproto import inet
from os_ken.tests.integrated import tester
def test_rule_set_tcp_src(self, dp):
    dl_type = ether.ETH_TYPE_IP
    ip_proto = inet.IPPROTO_TCP
    tp_src = 1103
    headers = [dp.ofproto.OXM_OF_TCP_SRC]
    self._set_verify(headers, tp_src)
    match = dp.ofproto_parser.OFPMatch()
    match.set_dl_type(dl_type)
    match.set_ip_proto(ip_proto)
    match.set_tcp_src(tp_src)
    self.add_matches(dp, match)