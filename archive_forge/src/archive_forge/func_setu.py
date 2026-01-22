from sympy.combinatorics.permutations import Permutation
from sympy.core.symbol import symbols
from sympy.matrices import Matrix
from sympy.utilities.iterables import variations, rotate_left
def setu(f, i, s):
    faces[f][i - 1, :] = Matrix(1, n, s)