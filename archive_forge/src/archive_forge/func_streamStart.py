from yowsup.structs import ProtocolTreeNode
import math
import binascii
import sys
import zlib
def streamStart(self, data):
    self.streamStarted = True
    tag = data.pop(0)
    size = self.readListSize(tag, data)
    tag = data.pop(0)
    if tag != 1:
        if tag == 236:
            tag = data.pop(0) + 237
        token = self.getToken(tag, data)
        raise Exception('expecting STREAM_START in streamStart, instead got token: %s' % token)
    attribCount = (size - 2 + size % 2) / 2
    self.readAttributes(attribCount, data)