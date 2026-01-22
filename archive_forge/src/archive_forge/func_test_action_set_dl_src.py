import logging
from os_ken.tests.integrated import tester
from os_ken.ofproto import ofproto_v1_0
from os_ken.ofproto import ether
from os_ken.ofproto import nx_match
def test_action_set_dl_src(self, dp):
    dl_src = '56:b3:42:04:b2:7a'
    dl_src_bin = self.haddr_to_bin(dl_src)
    self._verify = [dp.ofproto.OFPAT_SET_DL_SRC, 'dl_addr', dl_src_bin]
    action = dp.ofproto_parser.OFPActionSetDlSrc(dl_src_bin)
    self.add_action(dp, [action])