import re
def parsechar(self, rest):
    m = charRE.match(rest)
    if m is None:
        raise error('syntax error in AFM file: ' + repr(rest))
    things = []
    for fr, to in m.regs[1:]:
        things.append(rest[fr:to])
    charname = things[2]
    del things[2]
    charnum, width, l, b, r, t = (int(thing) for thing in things)
    self._chars[charname] = (charnum, width, (l, b, r, t))