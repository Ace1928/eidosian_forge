from fontTools.cffLib import maxStackLimit
@staticmethod
def vhcurveto(args):
    if len(args) < 4 or len(args) % 8 not in {0, 1, 4, 5}:
        raise ValueError(args)
    last_args = None
    if len(args) % 2 == 1:
        lastStraight = len(args) % 8 == 5
        args, last_args = (args[:-5], args[-5:])
    it = _everyN(args, 4)
    try:
        while True:
            args = next(it)
            yield ('rrcurveto', [0, args[0], args[1], args[2], args[3], 0])
            args = next(it)
            yield ('rrcurveto', [args[0], 0, args[1], args[2], 0, args[3]])
    except StopIteration:
        pass
    if last_args:
        args = last_args
        if lastStraight:
            yield ('rrcurveto', [0, args[0], args[1], args[2], args[3], args[4]])
        else:
            yield ('rrcurveto', [args[0], 0, args[1], args[2], args[4], args[3]])