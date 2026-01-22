from math import sqrt
from operator import itemgetter
import numpy as np
from ase.atom import Atom
from ase.atoms import Atoms
from ase.data import reference_states, atomic_numbers
from ase.lattice.cubic import FaceCenteredCubic
def fcc110(symbol, size, a=None, vacuum=None, orthogonal=True, periodic=False):
    """FCC(110) surface.

    Supported special adsorption sites: 'ontop', 'longbridge',
    'shortbridge', 'hollow'."""
    if not orthogonal:
        raise NotImplementedError("Can't do non-orthogonal cell yet!")
    return _surface(symbol, 'fcc', '110', size, a, None, vacuum, periodic=periodic, orthogonal=orthogonal)