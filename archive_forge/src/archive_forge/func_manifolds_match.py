import snappy
import spherogram
import spherogram.links.orthogonal
from nsnappytools import appears_hyperbolic
from sage.all import *
def manifolds_match(M0, M1):
    isoms = M0.is_isometric_to(M1, True)
    assert len(isoms) == 1
    return matches_peripheral_data(isoms[0])