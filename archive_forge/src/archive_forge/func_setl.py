from sympy.combinatorics.permutations import Permutation
from sympy.core.symbol import symbols
from sympy.matrices import Matrix
from sympy.utilities.iterables import variations, rotate_left
def setl(f, i, s):
    faces[f][:, i - 1] = Matrix(n, 1, s)