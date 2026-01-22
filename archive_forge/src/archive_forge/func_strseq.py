import types
def strseq(object, convert, join=joinseq):
    """Recursively walk a sequence, stringifying each element.

    """
    if type(object) in [list, tuple]:
        return join([strseq(_o, convert, join) for _o in object])
    else:
        return convert(object)