import unittest
import logging
import inspect
import struct
from os_ken.lib import addrconv
from os_ken.lib.packet import cfm
def test_serialize_with_loopback_message(self):
    self.setUp_loopback_message()
    self.test_serialize()
    data = bytearray()
    prev = None
    buf = self.ins.serialize(data, prev)
    loopback_message = cfm.loopback_message.parser(bytes(buf))
    self.assertEqual(repr(self.message), repr(loopback_message))