import sys ; sys.path.insert(0, '..')
import DNS
import socket
import unittest
def testIPaddrUnpacking(self):
    """ bin2addr should give known output for known input """
    for i, s in self.knownValues:
        result = DNS.Lib.bin2addr(s)
        self.assertEqual(i, result)