from __future__ import annotations
import warnings
from typing import TYPE_CHECKING
import numpy as np
def standardize_cell(cell: Cell, to_primitive=False, no_idealize=False, symprec=1e-05, angle_tolerance=-1.0):
    """Return standardized cell. When the search failed, ``None`` is returned.

    Parameters
    ----------
    cell, symprec, angle_tolerance:
        See the docstring of get_symmetry.
    to_primitive : bool
        If True, the standardized primitive cell is created.
    no_idealize : bool
        If True, it is disabled to idealize lengths and angles of basis vectors
        and positions of atoms according to crystal symmetry.

    Returns
    -------
    The standardized unit cell or primitive cell is returned by a tuple of
    (lattice, positions, numbers). If it fails, None is returned.

    Notes
    -----
    .. versionadded:: 1.8

    Now :func:`refine_cell` and :func:`find_primitive` are shorthands of
    this method with combinations of these options.
    About the default choice of the setting, see the documentation of ``hall_number``
    argument of :func:`get_symmetry_dataset`. More detailed explanation is
    shown in the spglib (C-API) document.
    """
    _set_no_error()
    lattice, _positions, _numbers, _ = _expand_cell(cell)
    num_atom = len(_positions)
    positions = np.zeros((num_atom * 4, 3), dtype='double', order='C')
    positions[:num_atom] = _positions
    numbers = np.zeros(num_atom * 4, dtype='intc')
    numbers[:num_atom] = _numbers
    num_atom_std = _spglib.standardize_cell(lattice, positions, numbers, num_atom, to_primitive * 1, no_idealize * 1, symprec, angle_tolerance)
    _set_error_message()
    if num_atom_std > 0:
        return (np.array(lattice.T, dtype='double', order='C'), np.array(positions[:num_atom_std], dtype='double', order='C'), np.array(numbers[:num_atom_std], dtype='intc'))
    else:
        return None