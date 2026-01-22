from fontTools.cffLib import maxStackLimit
def commandsToProgram(commands):
    """Takes a commands list as returned by programToCommands() and converts
    it back to a T2CharString program list."""
    program = []
    for op, args in commands:
        if any((isinstance(arg, list) for arg in args)):
            args = _flattenBlendArgs(args)
        program.extend(args)
        if op:
            program.append(op)
    return program