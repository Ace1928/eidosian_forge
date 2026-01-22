import logging
import unittest
from os_ken.lib.packet import ethernet
from os_ken.lib.packet import vxlan
def test_vni_to_bin(self):
    self.assertEqual(b'\x124V', vxlan.vni_to_bin(self.vni))