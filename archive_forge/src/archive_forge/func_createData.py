import re
import itertools
@staticmethod
def createData(version, errorCorrectLevel, dataList):
    rsBlocks = QRRSBlock.getRSBlocks(version, errorCorrectLevel)
    buffer = QRBitBuffer()
    for data in dataList:
        data.write(buffer, version)
    totalDataCount = 0
    for block in rsBlocks:
        totalDataCount += block.dataCount
    if buffer.getLengthInBits() > totalDataCount * 8:
        raise Exception('code length overflow. (%d > %d)' % (buffer.getLengthInBits(), totalDataCount * 8))
    if buffer.getLengthInBits() + 4 <= totalDataCount * 8:
        buffer.put(0, 4)
    while buffer.getLengthInBits() % 8 != 0:
        buffer.putBit(False)
    while True:
        if buffer.getLengthInBits() >= totalDataCount * 8:
            break
        buffer.put(QRCode.PAD0, 8)
        if buffer.getLengthInBits() >= totalDataCount * 8:
            break
        buffer.put(QRCode.PAD1, 8)
    return QRCode.createBytes(buffer, rsBlocks)