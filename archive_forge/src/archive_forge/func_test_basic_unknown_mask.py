import unittest
import os_ken.ofproto.ofproto_v1_3 as ofp
def test_basic_unknown_mask(self):
    user = ('field_100', ('aG9nZWhvZ2U=', 'ZnVnYWZ1Z2E='))
    on_wire = b'\x00\x00\xc9\x10hogehogefugafuga'
    self._test(user, on_wire, 4)