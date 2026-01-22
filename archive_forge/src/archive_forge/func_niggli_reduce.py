import numpy as np
def niggli_reduce(atoms):
    """Convert the supplied atoms object's unit cell into its
    maximally-reduced Niggli unit cell. Even if the unit cell is already
    maximally reduced, it will be converted into its unique Niggli unit cell.
    This will also wrap all atoms into the new unit cell.

    References:

    Niggli, P. "Krystallographische und strukturtheoretische Grundbegriffe.
    Handbuch der Experimentalphysik", 1928, Vol. 7, Part 1, 108-176.

    Krivy, I. and Gruber, B., "A Unified Algorithm for Determining the
    Reduced (Niggli) Cell", Acta Cryst. 1976, A32, 297-298.

    Grosse-Kunstleve, R.W.; Sauter, N. K.; and Adams, P. D. "Numerically
    stable algorithms for the computation of reduced unit cells", Acta Cryst.
    2004, A60, 1-6.
    """
    assert all(atoms.pbc), 'Can only reduce 3d periodic unit cells!'
    new_cell, op = niggli_reduce_cell(atoms.cell)
    update_cell_and_positions(atoms, new_cell, op)