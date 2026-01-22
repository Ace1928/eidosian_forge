import sys ; sys.path.insert(0, '..')
import DNS
import socket
import unittest
def checkPackResult(self, buf):
    if not hasattr(self, 'packerExpectedResult'):
        if self.__class__.__name__ != 'PackerTestCase':
            print('P***', self, repr(buf.getbuf()))
    else:
        return self.assertEqual(buf.getbuf(), self.packerExpectedResult)