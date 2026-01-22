from ..pari import pari
import fractions
def matrix_add(m1, m2):
    return [[c1 + c2 for c1, c2 in zip(r1, r2)] for r1, r2 in zip(m1, m2)]