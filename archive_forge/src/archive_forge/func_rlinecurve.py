from fontTools.cffLib import maxStackLimit
@staticmethod
def rlinecurve(args):
    if len(args) < 8 or len(args) % 2 != 0:
        raise ValueError(args)
    args, last_args = (args[:-6], args[-6:])
    for args in _everyN(args, 2):
        yield ('rlineto', args)
    yield ('rrcurveto', last_args)