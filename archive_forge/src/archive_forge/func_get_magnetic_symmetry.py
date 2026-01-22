from __future__ import annotations
import warnings
from typing import TYPE_CHECKING
import numpy as np
def get_magnetic_symmetry(cell: Cell, symprec=1e-05, angle_tolerance=-1.0, mag_symprec=-1.0, is_axial=None, with_time_reversal=True) -> dict | None:
    """Find magnetic symmetry operations from a crystal structure and site tensors.

    Parameters
    ----------
    cell : tuple
        Crystal structure given either in tuple.
        In the case given by a tuple, it has to follow the form below,

        (basis vectors, atomic points, types in integer numbers, ...)

        - basis vectors : array_like
            shape=(3, 3), order='C', dtype='double'

            .. code-block::

                [[a_x, a_y, a_z],
                [b_x, b_y, b_z],
                [c_x, c_y, c_z]]

        - atomic points : array_like
            shape=(num_atom, 3), order='C', dtype='double'

            Atomic position vectors with respect to basis vectors, i.e.,
            given in  fractional coordinates.
        - types : array_like
            shape=(num_atom, ), dtype='intc'

            Integer numbers to distinguish species.
        - magmoms:
            case-I: Scalar
                shape=(num_atom, ), dtype='double'

                Each atomic site has a scalar value. With is_magnetic=True,
                values are included in the symmetry search in a way of
                collinear magnetic moments.
            case-II: Vectors
                shape=(num_atom, 3), order='C', dtype='double'

                Each atomic site has a vector. With is_magnetic=True,
                vectors are included in the symmetry search in a way of
                non-collinear magnetic moments.
    symprec : float
        Symmetry search tolerance in the unit of length.
    angle_tolerance : float
        Symmetry search tolerance in the unit of angle deg.
        Normally it is not recommended to use this argument.
        See a bit more detail at :ref:`variables_angle_tolerance`.
        If the value is negative, an internally optimized routine is used to judge
        symmetry.
    mag_symprec : float
        Tolerance for magnetic symmetry search in the unit of magnetic moments.
        If not specified, use the same value as symprec.
    is_axial: None or bool
        Set `is_axial=True` if `magmoms` does not change their sign by improper
        rotations. If not specified, set `is_axial=False` when
        `magmoms.shape==(num_atoms, )`, and set `is_axial=True` when
        `magmoms.shape==(num_atoms, 3)`. These default settings correspond to
        collinear and non-collinear spins.
    with_time_reversal: bool
        Set `with_time_reversal=True` if `magmoms` change their sign by time-reversal
        operations. Default is True.

    Returns
    -------
    symmetry: dict or None
        Rotation parts and translation parts of symmetry operations represented
        with respect to basis vectors and atom index mapping by symmetry
        operations.

        - 'rotations' : ndarray
            shape=(num_operations, 3, 3), order='C', dtype='intc'

            Rotation (matrix) parts of symmetry operations
        - 'translations' : ndarray
            shape=(num_operations, 3), dtype='double'

            Translation (vector) parts of symmetry operations
        - 'time_reversals': ndarray
            shape=(num_operations, ), dtype='bool\\_'

            Time reversal part of magnetic symmetry operations.
            True indicates time reversal operation, and False indicates
            an ordinary operation.
        - 'equivalent_atoms' : ndarray
            shape=(num_atoms, ), dtype='intc'

    Notes
    -----
    .. versionadded:: 2.0
    """
    _set_no_error()
    lattice, positions, numbers, magmoms = _expand_cell(cell)
    if magmoms is None:
        raise TypeError('Specify magnetic moments in cell.')
    max_size = len(positions) * 96
    rotations = np.zeros((max_size, 3, 3), dtype='intc', order='C')
    translations = np.zeros((max_size, 3), dtype='double', order='C')
    equivalent_atoms = np.zeros(len(magmoms), dtype='intc')
    primitive_lattice = np.zeros((3, 3), dtype='double', order='C')
    if magmoms.ndim == 1 or magmoms.ndim == 2:
        spin_flips = np.zeros(max_size, dtype='intc')
    else:
        spin_flips = None
    if is_axial is None:
        if magmoms.ndim == 1:
            is_axial = False
        elif magmoms.ndim == 2:
            is_axial = True
    num_sym = _spglib.symmetry_with_site_tensors(rotations, translations, equivalent_atoms, primitive_lattice, spin_flips, lattice, positions, numbers, magmoms, with_time_reversal * 1, is_axial * 1, symprec, angle_tolerance, mag_symprec)
    _set_error_message()
    if num_sym == 0:
        return None
    else:
        spin_flips = np.array(spin_flips[:num_sym], dtype='intc', order='C')
        time_reversals = spin_flips == -1
        return {'rotations': np.array(rotations[:num_sym], dtype='intc', order='C'), 'translations': np.array(translations[:num_sym], dtype='double', order='C'), 'time_reversals': time_reversals, 'equivalent_atoms': equivalent_atoms, 'primitive_lattice': np.array(np.transpose(primitive_lattice), dtype='double', order='C')}