import re
def parsekernpair(self, rest):
    m = kernRE.match(rest)
    if m is None:
        raise error('syntax error in AFM file: ' + repr(rest))
    things = []
    for fr, to in m.regs[1:]:
        things.append(rest[fr:to])
    leftchar, rightchar, value = things
    value = int(value)
    self._kerning[leftchar, rightchar] = value