def writeInt20(self, v, data):
    data.append((983040 & v) >> 16)
    data.append((65280 & v) >> 8)
    data.append((v & 255) >> 0)