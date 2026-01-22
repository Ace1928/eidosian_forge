from sympy.utilities.iterables import kbins
def occur_check(var, x):
    """ var occurs in subtree owned by x? """
    if var == x:
        return True
    elif isinstance(x, Compound):
        return occur_check(var, x.args)
    elif is_args(x):
        if any((occur_check(var, xi) for xi in x)):
            return True
    return False