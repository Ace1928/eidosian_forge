from ..pari import pari
import fractions
def is_vector_zero(v):
    for e in v:
        if not e == 0:
            return False
    return True