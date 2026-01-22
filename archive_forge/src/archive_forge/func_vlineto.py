from fontTools.cffLib import maxStackLimit
@staticmethod
def vlineto(args):
    if not args:
        raise ValueError(args)
    it = iter(args)
    try:
        while True:
            yield ('rlineto', [0, next(it)])
            yield ('rlineto', [next(it), 0])
    except StopIteration:
        pass