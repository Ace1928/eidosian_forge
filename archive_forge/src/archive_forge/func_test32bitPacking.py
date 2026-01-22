import sys ; sys.path.insert(0, '..')
import DNS
import socket
import unittest
def test32bitPacking(self):
    """ pack32bit should give known output for known input """
    for i, s in self.knownValues:
        result = DNS.Lib.pack32bit(i)
        self.assertEqual(s, result)