import os
import warnings
from functools import total_ordering
from typing import Union
import numpy as np
def get_spacegroup(atoms, symprec=1e-05):
    """Determine the spacegroup to which belongs the Atoms object.

    This requires spglib: https://atztogo.github.io/spglib/ .

    Parameters:

    atoms: Atoms object
        Types, positions and unit-cell.
    symprec: float
        Symmetry tolerance, i.e. distance tolerance in Cartesian
        coordinates to find crystal symmetry.

    The Spacegroup object is returned.
    """
    import spglib
    sg = spglib.get_spacegroup((atoms.get_cell(), atoms.get_scaled_positions(), atoms.get_atomic_numbers()), symprec=symprec)
    if sg is None:
        raise RuntimeError('Spacegroup not found')
    sg_no = int(sg[sg.find('(') + 1:sg.find(')')])
    return Spacegroup(sg_no)