import re
from numpy import zeros, isscalar
from ase.atoms import Atoms
from ase.units import _auf, _amu, _auv
from ase.data import chemical_symbols
from ase.calculators.singlepoint import SinglePointCalculator
def iread_dlp_history(f, symbols=None):
    """Generator version of read_history"""
    levcfg, imcon, natoms, pos = _get_frame_positions(f)
    for p in pos:
        f.seek(p + 1)
        yield read_single_image(f, levcfg, imcon, natoms, is_trajectory=True, symbols=symbols)