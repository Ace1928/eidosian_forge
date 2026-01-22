import unittest
import os_ken.ofproto.ofproto_v1_3 as ofp
def test_exp_mask(self):
    user = ('_dp_hash', (305419896, 2147483647))
    on_wire = b'\xff\xff\x01\x0c\x00\x00# \x124Vx\x7f\xff\xff\xff'
    self._test(user, on_wire, 8)