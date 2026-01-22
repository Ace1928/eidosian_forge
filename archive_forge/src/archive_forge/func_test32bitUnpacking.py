import sys ; sys.path.insert(0, '..')
import DNS
import socket
import unittest
def test32bitUnpacking(self):
    """ unpack32bit should give known output for known input """
    for i, s in self.knownValues:
        result = DNS.Lib.unpack32bit(s)
        self.assertEqual(i, result)