import re
import itertools
@staticmethod
def getRSBlocks(version, errorCorrectLevel):
    rsBlock = QRRSBlock.getRsBlockTable(version, errorCorrectLevel)
    if rsBlock == None:
        raise Exception('bad rs block @ version:' + version + '/errorCorrectLevel:' + errorCorrectLevel)
    length = len(rsBlock) // 3
    list = []
    for i in range(length):
        count = rsBlock[i * 3 + 0]
        totalCount = rsBlock[i * 3 + 1]
        dataCount = rsBlock[i * 3 + 2]
        for j in range(count):
            list.append(QRRSBlock(totalCount, dataCount))
    return list