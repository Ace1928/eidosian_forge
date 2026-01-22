from fontTools.cffLib import maxStackLimit
@staticmethod
def vvcurveto(args):
    if len(args) < 4 or len(args) % 4 > 1:
        raise ValueError(args)
    if len(args) % 2 == 1:
        yield ('rrcurveto', [args[0], args[1], args[2], args[3], 0, args[4]])
        args = args[5:]
    for args in _everyN(args, 4):
        yield ('rrcurveto', [0, args[0], args[1], args[2], 0, args[3]])