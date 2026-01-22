import sys
def dictformat(_format, L={}, G={}):
    format = _format
    S = {}
    chunks = []
    pos = 0
    n = 0
    while 1:
        pc = format.find('%', pos)
        if pc < 0:
            break
        nextchar = format[pc + 1]
        if nextchar == '(':
            chunks.append(format[pos:pc])
            pos, level = (pc + 2, 1)
            while level:
                match, pos = _matchorfail(format, pos)
                tstart, tend = match.regs[3]
                token = format[tstart:tend]
                if token == '(':
                    level = level + 1
                elif token == ')':
                    level = level - 1
            vname = '__superformat_%d' % n
            n += 1
            S[vname] = eval(format[pc + 2:pos - 1], G, L)
            chunks.append('%%(%s)' % vname)
        else:
            nc = pc + 1 + (nextchar == '%')
            chunks.append(format[pos:nc])
            pos = nc
    if pos < len(format):
        chunks.append(format[pos:])
    return ''.join(chunks) % S