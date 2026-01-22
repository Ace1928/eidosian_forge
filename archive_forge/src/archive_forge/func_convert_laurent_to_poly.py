import string
from ..sage_helper import _within_sage, sage_method
def convert_laurent_to_poly(elt, minexp, P):
    if P.ngens() == 1:
        f = minexp[0]
        return P({e - f: c for e, c in elt.dict().items()})
    return P({tuple((e - f for e, f in zip(exps, minexp))): c for exps, c in elt.dict().items()})