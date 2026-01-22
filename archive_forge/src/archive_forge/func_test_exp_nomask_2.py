import unittest
import os_ken.ofproto.ofproto_v1_3 as ofp
def test_exp_nomask_2(self):
    user = ('tcp_flags', 2166)
    on_wire = b'\xff\xffT\x06ONF\x00\x08v'
    self._test(user, on_wire, 8)