import re
import itertools
def getBestMaskPattern(self):
    minLostPoint = 0
    pattern = 0
    for i in range(8):
        self.makeImpl(True, i)
        lostPoint = QRUtil.getLostPoint(self)
        if i == 0 or minLostPoint > lostPoint:
            minLostPoint = lostPoint
            pattern = i
    return pattern