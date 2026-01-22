from math import sqrt
from operator import itemgetter
import numpy as np
from ase.atom import Atom
from ase.atoms import Atoms
from ase.data import reference_states, atomic_numbers
from ase.lattice.cubic import FaceCenteredCubic
def bcc100(symbol, size, a=None, vacuum=None, orthogonal=True, periodic=False):
    """BCC(100) surface.

    Supported special adsorption sites: 'ontop', 'bridge', 'hollow'."""
    if not orthogonal:
        raise NotImplementedError("Can't do non-orthogonal cell yet!")
    return _surface(symbol, 'bcc', '100', size, a, None, vacuum, periodic=periodic, orthogonal=orthogonal)