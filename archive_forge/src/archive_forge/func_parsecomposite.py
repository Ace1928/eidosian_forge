import re
def parsecomposite(self, rest):
    m = compositeRE.match(rest)
    if m is None:
        raise error('syntax error in AFM file: ' + repr(rest))
    charname = m.group(1)
    ncomponents = int(m.group(2))
    rest = rest[m.regs[0][1]:]
    components = []
    while True:
        m = componentRE.match(rest)
        if m is None:
            raise error('syntax error in AFM file: ' + repr(rest))
        basechar = m.group(1)
        xoffset = int(m.group(2))
        yoffset = int(m.group(3))
        components.append((basechar, xoffset, yoffset))
        rest = rest[m.regs[0][1]:]
        if not rest:
            break
    assert len(components) == ncomponents
    self._composites[charname] = components