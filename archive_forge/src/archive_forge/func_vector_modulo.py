from ..pari import pari
import fractions
def vector_modulo(v, mod):
    return [x % mod for x in v]