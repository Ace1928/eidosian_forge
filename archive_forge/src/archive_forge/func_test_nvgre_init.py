import logging
import os
import sys
import unittest
from os_ken.lib import pcaplib
from os_ken.lib.packet import gre
from os_ken.lib.packet import packet
from os_ken.utils import binary_str
from os_ken.lib.packet.ether_types import ETH_TYPE_IP, ETH_TYPE_TEB
def test_nvgre_init(self):
    nvgre = gre.nvgre(version=self.version, vsid=self.vsid, flow_id=self.flow_id)
    self.assertEqual(nvgre.version, self.version)
    self.assertEqual(nvgre.protocol, self.nvgre_proto)
    self.assertEqual(nvgre.checksum, None)
    self.assertEqual(nvgre.seq_number, None)
    self.assertEqual(nvgre._key, self.key)
    self.assertEqual(nvgre._vsid, self.vsid)
    self.assertEqual(nvgre._flow_id, self.flow_id)