from fontTools.cffLib import maxStackLimit
@staticmethod
def hlineto(args):
    if not args:
        raise ValueError(args)
    it = iter(args)
    try:
        while True:
            yield ('rlineto', [next(it), 0])
            yield ('rlineto', [0, next(it)])
    except StopIteration:
        pass