import binascii
import re
import string
import struct
import types
from hashlib import md5, sha1, sha256, sha384, sha512
from typing import Dict, List, Optional, Tuple, Type
from twisted import __version__ as twisted_version
from twisted.conch.error import ConchError
from twisted.conch.ssh import _kex, address, service
from twisted.internet import defer
from twisted.protocols import loopback
from twisted.python import randbytes
from twisted.python.compat import iterbytes
from twisted.python.randbytes import insecureRandom
from twisted.python.reflect import requireModule
from twisted.test import proto_helpers
from twisted.trial.unittest import TestCase
def test_unimplementedPackets(self):
    """
        Test that unimplemented packet types cause MSG_UNIMPLEMENTED packets
        to be sent.
        """
    seqnum = self.proto.incomingPacketSequence

    def checkUnimplemented(seqnum=seqnum):
        self.assertEqual(self.packets[0][0], transport.MSG_UNIMPLEMENTED)
        self.assertEqual(self.packets[0][1][3:4], bytes((seqnum,)))
        self.proto.packets = []
        seqnum += 1
    self.proto.dispatchMessage(40, b'')
    checkUnimplemented()
    transport.messages[41] = b'MSG_fiction'
    self.proto.dispatchMessage(41, b'')
    checkUnimplemented()
    self.proto.dispatchMessage(60, b'')
    checkUnimplemented()
    self.proto.setService(MockService())
    self.proto.dispatchMessage(70, b'')
    checkUnimplemented()
    self.proto.dispatchMessage(71, b'')
    checkUnimplemented()