from yowsup.structs import ProtocolTreeNode
import math
import binascii
import sys
import zlib
def readNibble(self, data):
    _byte = self.readInt8(data)
    ignoreLastNibble = bool(_byte & 128)
    size = _byte & 127
    nrOfNibbles = size * 2 - int(ignoreLastNibble)
    dataArr = self.readArray(size, data)
    string = ''
    for i in range(0, nrOfNibbles):
        _byte = dataArr[int(math.floor(i / 2))]
        _shift = 4 * (1 - i % 2)
        dec = (_byte & 15 << _shift) >> _shift
        if dec in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9):
            string += str(dec)
        elif dec in (10, 11):
            string += chr(dec - 10 + 45)
        else:
            raise Exception('Bad nibble %s' % dec)
    return string