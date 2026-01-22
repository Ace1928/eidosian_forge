import unittest
import logging
import inspect
import struct
from os_ken.lib import addrconv
from os_ken.lib.packet import cfm
def test_serialize_with_loopback_reply(self):
    self.setUp_loopback_reply()
    self.test_serialize()
    data = bytearray()
    prev = None
    buf = self.ins.serialize(data, prev)
    loopback_reply = cfm.loopback_reply.parser(bytes(buf))
    self.assertEqual(repr(self.message), repr(loopback_reply))