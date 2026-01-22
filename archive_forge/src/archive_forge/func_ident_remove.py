from sympy.utilities.iterables import sift
from .util import new
def ident_remove(expr):
    """ Remove identities """
    ids = list(map(isid, expr.args))
    if sum(ids) == 0:
        return expr
    elif sum(ids) != len(ids):
        return new(expr.__class__, *[arg for arg, x in zip(expr.args, ids) if not x])
    else:
        return new(expr.__class__, expr.args[0])