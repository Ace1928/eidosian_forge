from math import sqrt
from operator import itemgetter
import numpy as np
from ase.atom import Atom
from ase.atoms import Atoms
from ase.data import reference_states, atomic_numbers
from ase.lattice.cubic import FaceCenteredCubic
def diamond111(symbol, size, a=None, vacuum=None, orthogonal=False, periodic=False):
    """DIAMOND(111) surface.

    Supported special adsorption sites: 'ontop'."""
    if orthogonal:
        raise NotImplementedError("Can't do orthogonal cell yet!")
    return _surface(symbol, 'diamond', '111', size, a, None, vacuum, periodic=periodic, orthogonal=orthogonal)