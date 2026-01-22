from ..pari import pari
import fractions
def max_abs_of_col(m, col):
    return max([abs(row[col]) for row in m])