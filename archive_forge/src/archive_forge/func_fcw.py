from sympy.combinatorics.permutations import Permutation
from sympy.core.symbol import symbols
from sympy.matrices import Matrix
from sympy.utilities.iterables import variations, rotate_left
def fcw(i, r=1):
    for _ in range(r):
        if i == 0:
            cw(F)
        i += 1
        temp = getr(L, i)
        setr(L, i, list(getu(D, i)))
        setu(D, i, list(reversed(getl(R, i))))
        setl(R, i, list(getd(U, i)))
        setd(U, i, list(reversed(temp)))
        i -= 1