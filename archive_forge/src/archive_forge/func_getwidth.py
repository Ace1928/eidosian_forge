from ._constants import *
def getwidth(self):
    if self.width is not None:
        return self.width
    lo = hi = 0
    for op, av in self.data:
        if op is BRANCH:
            i = MAXWIDTH
            j = 0
            for av in av[1]:
                l, h = av.getwidth()
                i = min(i, l)
                j = max(j, h)
            lo = lo + i
            hi = hi + j
        elif op is ATOMIC_GROUP:
            i, j = av.getwidth()
            lo = lo + i
            hi = hi + j
        elif op is SUBPATTERN:
            i, j = av[-1].getwidth()
            lo = lo + i
            hi = hi + j
        elif op in _REPEATCODES:
            i, j = av[2].getwidth()
            lo = lo + i * av[0]
            if av[1] == MAXREPEAT and j:
                hi = MAXWIDTH
            else:
                hi = hi + j * av[1]
        elif op in _UNITCODES:
            lo = lo + 1
            hi = hi + 1
        elif op is GROUPREF:
            i, j = self.state.groupwidths[av]
            lo = lo + i
            hi = hi + j
        elif op is GROUPREF_EXISTS:
            i, j = av[1].getwidth()
            if av[2] is not None:
                l, h = av[2].getwidth()
                i = min(i, l)
                j = max(j, h)
            else:
                i = 0
            lo = lo + i
            hi = hi + j
        elif op is SUCCESS:
            break
    self.width = (min(lo, MAXWIDTH), min(hi, MAXWIDTH))
    return self.width