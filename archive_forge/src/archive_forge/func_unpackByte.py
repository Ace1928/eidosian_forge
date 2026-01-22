from yowsup.structs import ProtocolTreeNode
import math
import binascii
import sys
import zlib
def unpackByte(self, n, n2):
    if n == 251:
        return self.unpackHex(n2)
    if n == 255:
        return self.unpackNibble(n2)
    raise ValueError('bad packed type %s' % n)