from __future__ import annotations
import warnings
from typing import TYPE_CHECKING
import numpy as np
def relocate_BZ_grid_address(grid_address, mesh, reciprocal_lattice, is_shift=None, is_dense=False):
    """Grid addresses are relocated to be inside first Brillouin zone.

    Number of ir-grid-points inside Brillouin zone is returned.
    It is assumed that the following arrays have the shapes of
        bz_grid_address : (num_grid_points_in_FBZ, 3)
        bz_map (prod(mesh * 2), )

    Note that the shape of grid_address is (prod(mesh), 3) and the
    addresses in grid_address are arranged to be in parallelepiped
    made of reciprocal basis vectors. The addresses in bz_grid_address
    are inside the first Brillouin zone or on its surface. Each
    address in grid_address is mapped to one of those in
    bz_grid_address by a reciprocal lattice vector (including zero
    vector) with keeping element order. For those inside first
    Brillouin zone, the mapping is one-to-one. For those on the first
    Brillouin zone surface, more than one addresses in bz_grid_address
    that are equivalent by the reciprocal lattice translations are
    mapped to one address in grid_address. In this case, those grid
    points except for one of them are appended to the tail of this array,
    for which bz_grid_address has the following data storing:

    .. code-block::

      |------------------array size of bz_grid_address-------------------------|
      |--those equivalent to grid_address--|--those on surface except for one--|
      |-----array size of grid_address-----|

    Number of grid points stored in bz_grid_address is returned.
    bz_map is used to recover grid point index expanded to include BZ
    surface from grid address. The grid point indices are mapped to
    (mesh[0] * 2) x (mesh[1] * 2) x (mesh[2] * 2) space (bz_map).

    """
    _set_no_error()
    if is_shift is None:
        _is_shift = np.zeros(3, dtype='intc')
    else:
        _is_shift = np.array(is_shift, dtype='intc')
    bz_grid_address = np.zeros((np.prod(np.add(mesh, 1)), 3), dtype='intc')
    bz_map = np.zeros(np.prod(np.multiply(mesh, 2)), dtype='uintp')
    num_bz_ir = _spglib.BZ_grid_address(bz_grid_address, bz_map, grid_address, np.array(mesh, dtype='intc'), np.array(reciprocal_lattice, dtype='double', order='C'), _is_shift)
    if is_dense:
        return (bz_grid_address[:num_bz_ir], bz_map)
    else:
        return (bz_grid_address[:num_bz_ir], np.array(bz_map, dtype='intc'))