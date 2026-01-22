import inspect
import logging
import struct
import unittest
from os_ken.lib.packet import icmp
from os_ken.lib.packet import packet_utils
def test_json_with_dest_unreach(self):
    self.setUp_with_dest_unreach()
    self.test_json()