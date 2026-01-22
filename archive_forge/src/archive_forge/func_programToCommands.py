from fontTools.cffLib import maxStackLimit
def programToCommands(program, getNumRegions=None):
    """Takes a T2CharString program list and returns list of commands.
    Each command is a two-tuple of commandname,arg-list.  The commandname might
    be empty string if no commandname shall be emitted (used for glyph width,
    hintmask/cntrmask argument, as well as stray arguments at the end of the
    program (🤷).
    'getNumRegions' may be None, or a callable object. It must return the
    number of regions. 'getNumRegions' takes a single argument, vsindex. If
    the vsindex argument is None, getNumRegions returns the default number
    of regions for the charstring, else it returns the numRegions for
    the vsindex.
    The Charstring may or may not start with a width value. If the first
    non-blend operator has an odd number of arguments, then the first argument is
    a width, and is popped off. This is complicated with blend operators, as
    there may be more than one before the first hint or moveto operator, and each
    one reduces several arguments to just one list argument. We have to sum the
    number of arguments that are not part of the blend arguments, and all the
    'numBlends' values. We could instead have said that by definition, if there
    is a blend operator, there is no width value, since CFF2 Charstrings don't
    have width values. I discussed this with Behdad, and we are allowing for an
    initial width value in this case because developers may assemble a CFF2
    charstring from CFF Charstrings, which could have width values.
    """
    seenWidthOp = False
    vsIndex = None
    lenBlendStack = 0
    lastBlendIndex = 0
    commands = []
    stack = []
    it = iter(program)
    for token in it:
        if not isinstance(token, str):
            stack.append(token)
            continue
        if token == 'blend':
            assert getNumRegions is not None
            numSourceFonts = 1 + getNumRegions(vsIndex)
            numBlends = stack[-1]
            numBlendArgs = numBlends * numSourceFonts + 1
            stack[-numBlendArgs:] = [stack[-numBlendArgs:]]
            lenBlendStack += numBlends + len(stack) - 1
            lastBlendIndex = len(stack)
            continue
        elif token == 'vsindex':
            vsIndex = stack[-1]
            assert type(vsIndex) is int
        elif not seenWidthOp and token in {'hstem', 'hstemhm', 'vstem', 'vstemhm', 'cntrmask', 'hintmask', 'hmoveto', 'vmoveto', 'rmoveto', 'endchar'}:
            seenWidthOp = True
            parity = token in {'hmoveto', 'vmoveto'}
            if lenBlendStack:
                numArgs = lenBlendStack + len(stack[lastBlendIndex:])
            else:
                numArgs = len(stack)
            if numArgs and numArgs % 2 ^ parity:
                width = stack.pop(0)
                commands.append(('', [width]))
        if token in {'hintmask', 'cntrmask'}:
            if stack:
                commands.append(('', stack))
            commands.append((token, []))
            commands.append(('', [next(it)]))
        else:
            commands.append((token, stack))
        stack = []
    if stack:
        commands.append(('', stack))
    return commands