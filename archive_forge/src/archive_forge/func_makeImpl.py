import re
import itertools
def makeImpl(self, test, maskPattern):
    self.moduleCount = self.version * 4 + 17
    self.modules = [[False] * self.moduleCount for x in range(self.moduleCount)]
    self.setupPositionProbePattern(0, 0)
    self.setupPositionProbePattern(self.moduleCount - 7, 0)
    self.setupPositionProbePattern(0, self.moduleCount - 7)
    self.setupPositionAdjustPattern()
    self.setupTimingPattern()
    self.setupTypeInfo(test, maskPattern)
    if self.version >= 7:
        self.setupTypeNumber(test)
    if self.dataCache == None:
        self.dataCache = QRCode.createData(self.version, self.errorCorrectLevel, self.dataList)
    self.mapData(self.dataCache, maskPattern)