import logging
import unittest
from os_ken.lib.packet import ethernet
from os_ken.lib.packet import vxlan
def test_vni_from_bin(self):
    vni = vxlan.vni_from_bin(b'\x124V')
    self.assertEqual(self.vni, vni)