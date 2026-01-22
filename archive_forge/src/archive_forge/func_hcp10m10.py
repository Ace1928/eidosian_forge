from math import sqrt
from operator import itemgetter
import numpy as np
from ase.atom import Atom
from ase.atoms import Atoms
from ase.data import reference_states, atomic_numbers
from ase.lattice.cubic import FaceCenteredCubic
def hcp10m10(symbol, size, a=None, c=None, vacuum=None, orthogonal=True, periodic=False):
    """HCP(10m10) surface.

    Supported special adsorption sites: 'ontop'.

    Works only for size=(i,j,k) with j even."""
    if not orthogonal:
        raise NotImplementedError("Can't do non-orthogonal cell yet!")
    return _surface(symbol, 'hcp', '10m10', size, a, c, vacuum, periodic=periodic, orthogonal=orthogonal)