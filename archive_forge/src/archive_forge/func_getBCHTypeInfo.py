import re
import itertools
@staticmethod
def getBCHTypeInfo(data):
    d = data << 10
    while QRUtil.getBCHDigit(d) - QRUtil.getBCHDigit(QRUtil.G15) >= 0:
        d ^= QRUtil.G15 << QRUtil.getBCHDigit(d) - QRUtil.getBCHDigit(QRUtil.G15)
    return (data << 10 | d) ^ QRUtil.G15_MASK