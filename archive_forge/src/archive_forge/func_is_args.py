from sympy.utilities.iterables import kbins
def is_args(x):
    """ Is x a traditional iterable? """
    return type(x) in (tuple, list, set)