import logging
from os_ken.tests.integrated import tester
from os_ken.ofproto import ofproto_v1_0
from os_ken.ofproto import ether
from os_ken.ofproto import nx_match
def test_rule_set_dl_src(self, dp):
    dl_src = 'b8:a1:94:51:78:83'
    dl_src_bin = self.haddr_to_bin(dl_src)
    self._verify = ['dl_src', dl_src_bin]
    rule = nx_match.ClsRule()
    rule.set_dl_src(dl_src_bin)
    self.add_rule(dp, rule)