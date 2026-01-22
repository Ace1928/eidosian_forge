import re
from numpy import zeros, isscalar
from ase.atoms import Atoms
from ase.units import _auf, _amu, _auv
from ase.data import chemical_symbols
from ase.calculators.singlepoint import SinglePointCalculator
def read_dlp_history(f, index=-1, symbols=None):
    """Read a HISTORY file.

    Compatible with DLP4 and DLP_CLASSIC.

    *Index* can be integer or slice.

    Provide a list of element strings to symbols to ignore naming
    from the HISTORY file.
    """
    levcfg, imcon, natoms, pos = _get_frame_positions(f)
    if isscalar(index):
        selected = [pos[index]]
    else:
        selected = pos[index]
    images = []
    for fpos in selected:
        f.seek(fpos + 1)
        images.append(read_single_image(f, levcfg, imcon, natoms, is_trajectory=True, symbols=symbols))
    return images