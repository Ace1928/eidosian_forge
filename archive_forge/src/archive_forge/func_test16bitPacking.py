import sys ; sys.path.insert(0, '..')
import DNS
import socket
import unittest
def test16bitPacking(self):
    """ pack16bit should give known output for known input """
    for i, s in self.knownValues:
        result = DNS.Lib.pack16bit(i)
        self.assertEqual(s, result)