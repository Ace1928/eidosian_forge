import re
def updatepos(self, i, j):
    if i >= j:
        return j
    rawdata = self.rawdata
    nlines = rawdata.count('\n', i, j)
    if nlines:
        self.lineno = self.lineno + nlines
        pos = rawdata.rindex('\n', i, j)
        self.offset = j - (pos + 1)
    else:
        self.offset = self.offset + j - i
    return j