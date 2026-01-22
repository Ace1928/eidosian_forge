from fontTools.cffLib import maxStackLimit
@staticmethod
def rmoveto(args):
    if len(args) != 2:
        raise ValueError(args)
    yield ('rmoveto', args)