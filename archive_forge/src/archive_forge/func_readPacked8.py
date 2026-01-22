from yowsup.structs import ProtocolTreeNode
import math
import binascii
import sys
import zlib
def readPacked8(self, n, data):
    size = self.readInt8(data)
    remove = 0
    if size & 128 != 0 and n == 251:
        remove = 1
    size = size & 127
    text = bytearray(self.readArray(size, data))
    hexData = binascii.hexlify(str(text) if sys.version_info < (2, 7) else text).upper()
    dataSize = len(hexData)
    out = []
    if remove == 0:
        for i in range(0, dataSize):
            char = chr(hexData[i]) if type(hexData[i]) is int else hexData[i]
            val = ord(binascii.unhexlify('0%s' % char))
            if i == dataSize - 1 and val > 11 and (n != 251):
                continue
            out.append(self.unpackByte(n, val))
    else:
        out = map(ord, list(hexData[0:-remove])) if sys.version_info < (3, 0) else list(hexData[0:-remove])
    return out