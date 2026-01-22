from collections import Counter, defaultdict, OrderedDict
from itertools import (
from itertools import product as cartes # noqa: F401
from operator import gt
from sympy.utilities.enumerative import (
from sympy.utilities.misc import as_int
from sympy.utilities.decorator import deprecated
def finish_derangements():
    """Place the last two elements into the partially completed
        derangement, and yield the results.
        """
    a = take[1][0]
    a_ct = take[1][1]
    b = take[0][0]
    b_ct = take[0][1]
    forced_a = []
    forced_b = []
    open_free = []
    for i in range(len(s)):
        if rv[i] is None:
            if s[i] == a:
                forced_b.append(i)
            elif s[i] == b:
                forced_a.append(i)
            else:
                open_free.append(i)
    if len(forced_a) > a_ct or len(forced_b) > b_ct:
        return
    for i in forced_a:
        rv[i] = a
    for i in forced_b:
        rv[i] = b
    for a_place in combinations(open_free, a_ct - len(forced_a)):
        for a_pos in a_place:
            rv[a_pos] = a
        for i in open_free:
            if rv[i] is None:
                rv[i] = b
        yield rv
        for i in open_free:
            rv[i] = None
    for i in forced_a:
        rv[i] = None
    for i in forced_b:
        rv[i] = None