import sys ; sys.path.insert(0, '..')
import DNS
import socket
import unittest
def checkUnpackResult(self, rrbits, specbits):
    if not hasattr(self, 'unpackerExpectedResult'):
        if self.__class__.__name__ != 'PackerTestCase':
            print('U***', self, repr((rrbits, specbits)))
    else:
        return self.assertEqual((rrbits, specbits), self.unpackerExpectedResult)