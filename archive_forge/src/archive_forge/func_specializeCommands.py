from fontTools.cffLib import maxStackLimit
def specializeCommands(commands, ignoreErrors=False, generalizeFirst=True, preserveTopology=False, maxstack=48):
    if generalizeFirst:
        commands = generalizeCommands(commands, ignoreErrors=ignoreErrors)
    else:
        commands = list(commands)
    for i in range(len(commands) - 1, 0, -1):
        if 'rmoveto' == commands[i][0] == commands[i - 1][0]:
            v1, v2 = (commands[i - 1][1], commands[i][1])
            commands[i - 1] = ('rmoveto', [v1[0] + v2[0], v1[1] + v2[1]])
            del commands[i]
    for i in range(len(commands)):
        op, args = commands[i]
        if op in {'rmoveto', 'rlineto'}:
            c, args = _categorizeVector(args)
            commands[i] = (c + op[1:], args)
            continue
        if op == 'rrcurveto':
            c1, args1 = _categorizeVector(args[:2])
            c2, args2 = _categorizeVector(args[-2:])
            commands[i] = (c1 + c2 + 'curveto', args1 + args[2:4] + args2)
            continue
    if not preserveTopology:
        for i in range(len(commands) - 1, -1, -1):
            op, args = commands[i]
            if op == '00curveto':
                assert len(args) == 4
                c, args = _categorizeVector(args[1:3])
                op = c + 'lineto'
                commands[i] = (op, args)
            if op == '0lineto':
                del commands[i]
                continue
            if i and op in {'hlineto', 'vlineto'} and (op == commands[i - 1][0]):
                _, other_args = commands[i - 1]
                assert len(args) == 1 and len(other_args) == 1
                try:
                    new_args = [_addArgs(args[0], other_args[0])]
                except ValueError:
                    continue
                commands[i - 1] = (op, new_args)
                del commands[i]
                continue
    for i in range(1, len(commands) - 1):
        op, args = commands[i]
        prv, nxt = (commands[i - 1][0], commands[i + 1][0])
        if op in {'0lineto', 'hlineto', 'vlineto'} and prv == nxt == 'rlineto':
            assert len(args) == 1
            args = [0, args[0]] if op[0] == 'v' else [args[0], 0]
            commands[i] = ('rlineto', args)
            continue
        if op[2:] == 'curveto' and len(args) == 5 and (prv == nxt == 'rrcurveto'):
            assert (op[0] == 'r') ^ (op[1] == 'r')
            if op[0] == 'v':
                pos = 0
            elif op[0] != 'r':
                pos = 1
            elif op[1] == 'v':
                pos = 4
            else:
                pos = 5
            args = args[:pos] + type(args)((0,)) + args[pos:]
            commands[i] = ('rrcurveto', args)
            continue
    for i in range(len(commands) - 1, 0, -1):
        op1, args1 = commands[i - 1]
        op2, args2 = commands[i]
        new_op = None
        if {op1, op2} <= {'rlineto', 'rrcurveto'}:
            if op1 == op2:
                new_op = op1
            elif op2 == 'rrcurveto' and len(args2) == 6:
                new_op = 'rlinecurve'
            elif len(args2) == 2:
                new_op = 'rcurveline'
        elif (op1, op2) in {('rlineto', 'rlinecurve'), ('rrcurveto', 'rcurveline')}:
            new_op = op2
        elif {op1, op2} == {'vlineto', 'hlineto'}:
            new_op = op1
        elif 'curveto' == op1[2:] == op2[2:]:
            d0, d1 = op1[:2]
            d2, d3 = op2[:2]
            if d1 == 'r' or d2 == 'r' or d0 == d3 == 'r':
                continue
            d = _mergeCategories(d1, d2)
            if d is None:
                continue
            if d0 == 'r':
                d = _mergeCategories(d, d3)
                if d is None:
                    continue
                new_op = 'r' + d + 'curveto'
            elif d3 == 'r':
                d0 = _mergeCategories(d0, _negateCategory(d))
                if d0 is None:
                    continue
                new_op = d0 + 'r' + 'curveto'
            else:
                d0 = _mergeCategories(d0, d3)
                if d0 is None:
                    continue
                new_op = d0 + d + 'curveto'
        if new_op and len(args1) + len(args2) < maxstack:
            commands[i - 1] = (new_op, args1 + args2)
            del commands[i]
    for i in range(len(commands)):
        op, args = commands[i]
        if op in {'0moveto', '0lineto'}:
            commands[i] = ('h' + op[1:], args)
            continue
        if op[2:] == 'curveto' and op[:2] not in {'rr', 'hh', 'vv', 'vh', 'hv'}:
            op0, op1 = op[:2]
            if (op0 == 'r') ^ (op1 == 'r'):
                assert len(args) % 2 == 1
            if op0 == '0':
                op0 = 'h'
            if op1 == '0':
                op1 = 'h'
            if op0 == 'r':
                op0 = op1
            if op1 == 'r':
                op1 = _negateCategory(op0)
            assert {op0, op1} <= {'h', 'v'}, (op0, op1)
            if len(args) % 2:
                if op0 != op1:
                    if (op0 == 'h') ^ (len(args) % 8 == 1):
                        args = args[:-2] + args[-1:] + args[-2:-1]
                elif op0 == 'h':
                    args = args[1:2] + args[:1] + args[2:]
            commands[i] = (op0 + op1 + 'curveto', args)
            continue
    for i in range(len(commands)):
        op, args = commands[i]
        if any((isinstance(arg, list) for arg in args)):
            commands[i] = (op, _convertToBlendCmds(args))
    return commands