from collections import Counter, defaultdict, OrderedDict
from itertools import (
from itertools import product as cartes # noqa: F401
from operator import gt
from sympy.utilities.enumerative import (
from sympy.utilities.misc import as_int
from sympy.utilities.decorator import deprecated
def multiset_derangements(s):
    """Generate derangements of the elements of s *in place*.

    Examples
    ========

    >>> from sympy.utilities.iterables import multiset_derangements, uniq

    Because the derangements of multisets (not sets) are generated
    in place, copies of the return value must be made if a collection
    of derangements is desired or else all values will be the same:

    >>> list(uniq([i for i in multiset_derangements('1233')]))
    [[None, None, None, None]]
    >>> [i.copy() for i in multiset_derangements('1233')]
    [['3', '3', '1', '2'], ['3', '3', '2', '1']]
    >>> [''.join(i) for i in multiset_derangements('1233')]
    ['3312', '3321']
    """
    from sympy.core.sorting import ordered
    try:
        ms = multiset(s)
    except TypeError:
        key = dict(enumerate(ordered(uniq(s))))
        h = []
        for si in s:
            for k in key:
                if key[k] == si:
                    h.append(k)
                    break
        for i in multiset_derangements(h):
            yield [key[j] for j in i]
        return
    mx = max(ms.values())
    n = len(s)
    if mx * 2 > n:
        return
    if len(ms) == n:
        yield from _set_derangements(s)
        return
    for M in ms:
        if ms[M] == mx:
            break
    inonM = [i for i in range(n) if s[i] != M]
    iM = [i for i in range(n) if s[i] == M]
    rv = [None] * n
    if 2 * mx == n:
        for i in inonM:
            rv[i] = M
        for p in multiset_permutations([s[i] for i in inonM]):
            for i, pi in zip(iM, p):
                rv[i] = pi
            yield rv
        rv[:] = [None] * n
        return
    if n - 2 * mx == 1 and len(ms.values()) == n - mx + 1:
        for i, i1 in enumerate(inonM):
            ifill = inonM[:i] + inonM[i + 1:]
            for j in ifill:
                rv[j] = M
            for p in permutations([s[j] for j in ifill]):
                rv[i1] = s[i1]
                for j, pi in zip(iM, p):
                    rv[j] = pi
                k = i1
                for j in iM:
                    rv[j], rv[k] = (rv[k], rv[j])
                    yield rv
                    k = j
        rv[:] = [None] * n
        return

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

    def iopen(v):
        return [i for i in range(n) if rv[i] is None and s[i] != v]

    def do(j):
        if j == 1:
            yield from finish_derangements()
        else:
            M, mx = take[j]
            for i in combinations(iopen(M), mx):
                for ii in i:
                    rv[ii] = M
                yield from do(j - 1)
                for ii in i:
                    rv[ii] = None
    take = sorted(ms.items(), key=lambda x: (x[1], x[0]))
    yield from do(len(take) - 1)
    rv[:] = [None] * n