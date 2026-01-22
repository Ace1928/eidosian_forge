from yowsup.structs import ProtocolTreeNode
import math
import binascii
import sys
import zlib
def getTokenDouble(self, n, n2):
    pos = n2 + n * 256
    token = self.tokenDictionary.getToken(pos, True)
    if not token:
        raise ValueError('Invalid token %s' % pos)
    return token