from math import sqrt
from operator import itemgetter
import numpy as np
from ase.atom import Atom
from ase.atoms import Atoms
from ase.data import reference_states, atomic_numbers
from ase.lattice.cubic import FaceCenteredCubic
def bcc110(symbol, size, a=None, vacuum=None, orthogonal=False, periodic=False):
    """BCC(110) surface.

    Supported special adsorption sites: 'ontop', 'longbridge',
    'shortbridge', 'hollow'.

    Use *orthogonal=True* to get an orthogonal unit cell - works only
    for size=(i,j,k) with j even."""
    return _surface(symbol, 'bcc', '110', size, a, None, vacuum, periodic=periodic, orthogonal=orthogonal)