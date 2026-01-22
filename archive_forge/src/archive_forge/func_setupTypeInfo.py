import re
import itertools
def setupTypeInfo(self, test, maskPattern):
    data = self.errorCorrectLevel << 3 | maskPattern
    bits = QRUtil.getBCHTypeInfo(data)
    for i in range(15):
        mod = not test and bits >> i & 1 == 1
        if i < 6:
            self.modules[i][8] = mod
        elif i < 8:
            self.modules[i + 1][8] = mod
        else:
            self.modules[self.moduleCount - 15 + i][8] = mod
    for i in range(15):
        mod = not test and bits >> i & 1 == 1
        if i < 8:
            self.modules[8][self.moduleCount - i - 1] = mod
        elif i < 9:
            self.modules[8][15 - i - 1 + 1] = mod
        else:
            self.modules[8][15 - i - 1] = mod
    self.modules[self.moduleCount - 8][8] = not test