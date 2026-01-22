from io import StringIO
from antlr4.Token import Token
def removeOne(self, v):
    if self.intervals is not None:
        k = 0
        for i in self.intervals:
            if v < i.start:
                return
            elif v == i.start and v == i.stop - 1:
                self.intervals.pop(k)
                return
            elif v == i.start:
                self.intervals[k] = range(i.start + 1, i.stop)
                return
            elif v == i.stop - 1:
                self.intervals[k] = range(i.start, i.stop - 1)
                return
            elif v < i.stop - 1:
                x = range(i.start, v)
                self.intervals[k] = range(v + 1, i.stop)
                self.intervals.insert(k, x)
                return
            k += 1