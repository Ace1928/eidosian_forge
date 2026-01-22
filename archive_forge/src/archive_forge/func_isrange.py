from pythran.passmanager import Transformation
from pythran.utils import isnum
import gast as ast
def isrange(self, elts):
    if not elts:
        return None
    if not all((isnum(x) and isinstance(x.value, int) for x in elts)):
        return None
    unboxed_ints = [x.value for x in elts]
    start = unboxed_ints[0]
    if len(unboxed_ints) == 1:
        return (start, start + 1, 1)
    else:
        step = unboxed_ints[1] - start
        stop = unboxed_ints[-1] + step
        if unboxed_ints == list(range(start, stop, step)):
            return (start, stop, step)
        else:
            return None