from __future__ import annotations
from ..runtime import driver
def getGlobalDims(self, args):
    shape = []
    for e in self.globalDimsArgIdx:
        t = 1
        if e == -1:
            t = 1
        elif e < 0 and e != -1:
            t = -e - 1
        else:
            idx = self.getOriginArgIdx(e, args)
            t = args[idx]
        shape.append(t)
    return shape