from yowsup.structs import ProtocolTreeNode
import math
import binascii
import sys
import zlib
def unpackHex(self, n):
    if n in range(0, 10):
        return n + 48
    if n in range(10, 16):
        return 65 + (n - 10)
    raise ValueError('bad hex %s' % n)