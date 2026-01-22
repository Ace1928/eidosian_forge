from fontTools.cffLib import maxStackLimit
@staticmethod
def vmoveto(args):
    if len(args) != 1:
        raise ValueError(args)
    yield ('rmoveto', [0, args[0]])