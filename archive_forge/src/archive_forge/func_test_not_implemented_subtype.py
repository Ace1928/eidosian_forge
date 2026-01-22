import copy
import logging
from struct import pack, unpack_from
import unittest
from os_ken.ofproto import ether
from os_ken.lib.packet.ethernet import ethernet
from os_ken.lib.packet.packet import Packet
from os_ken.lib import addrconv
from os_ken.lib.packet.slow import slow, lacp
from os_ken.lib.packet.slow import SLOW_PROTOCOL_MULTICAST
from os_ken.lib.packet.slow import SLOW_SUBTYPE_LACP
from os_ken.lib.packet.slow import SLOW_SUBTYPE_MARKER
def test_not_implemented_subtype(self):
    not_implemented_buf = pack(slow._PACK_STR, SLOW_SUBTYPE_MARKER) + self.buf[1:]
    instance, nexttype, last = slow.parser(not_implemented_buf)
    assert instance is None
    assert nexttype is None
    assert last is not None