import numpy as np
def minimize_tilt(atoms, order=range(3), fold_atoms=True):
    """Minimize the tilt angles of the unit cell."""
    pbc_c = atoms.get_pbc()
    for i1, c1 in enumerate(order):
        for c2 in order[i1 + 1:]:
            if pbc_c[c1] and pbc_c[c2]:
                minimize_tilt_ij(atoms, c1, c2, fold_atoms)