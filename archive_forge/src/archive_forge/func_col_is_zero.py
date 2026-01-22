from ..pari import pari
import fractions
def col_is_zero(m, col):
    if col < 0:
        return True
    for row in m:
        if not row[col] == 0:
            return False
    return True