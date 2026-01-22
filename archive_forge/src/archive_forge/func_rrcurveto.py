from fontTools.cffLib import maxStackLimit
@staticmethod
def rrcurveto(args):
    if not args:
        raise ValueError(args)
    for args in _everyN(args, 6):
        yield ('rrcurveto', args)