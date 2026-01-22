from fontTools.cffLib import maxStackLimit
@staticmethod
def rcurveline(args):
    if len(args) < 8 or len(args) % 6 != 2:
        raise ValueError(args)
    args, last_args = (args[:-2], args[-2:])
    for args in _everyN(args, 6):
        yield ('rrcurveto', args)
    yield ('rlineto', last_args)