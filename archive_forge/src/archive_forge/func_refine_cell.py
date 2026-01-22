from __future__ import annotations
import warnings
from typing import TYPE_CHECKING
import numpy as np
def refine_cell(cell: Cell, symprec=1e-05, angle_tolerance=-1.0):
    """Return refined cell. When the search failed, ``None`` is returned.

    The standardized unit cell is returned by a tuple of
    (lattice, positions, numbers).

    Notes
    -----
    .. versionchanged:: 1.8

    The detailed control of standardization of unit cell can be done using
    :func:`standardize_cell`.
    """
    _set_no_error()
    lattice, _positions, _numbers, _ = _expand_cell(cell)
    num_atom = len(_positions)
    positions = np.zeros((num_atom * 4, 3), dtype='double', order='C')
    positions[:num_atom] = _positions
    numbers = np.zeros(num_atom * 4, dtype='intc')
    numbers[:num_atom] = _numbers
    num_atom_std = _spglib.refine_cell(lattice, positions, numbers, num_atom, symprec, angle_tolerance)
    _set_error_message()
    if num_atom_std > 0:
        return (np.array(lattice.T, dtype='double', order='C'), np.array(positions[:num_atom_std], dtype='double', order='C'), np.array(numbers[:num_atom_std], dtype='intc'))
    else:
        return None