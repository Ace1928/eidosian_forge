import numpy as np
def minimize_tilt_ij(atoms, modified=1, fixed=0, fold_atoms=True):
    """Minimize the tilt angle for two given axes.

    The problem is underdetermined. Therefore one can choose one axis
    that is kept fixed.
    """
    orgcell_cc = atoms.get_cell()
    pbc_c = atoms.get_pbc()
    i = fixed
    j = modified
    if not (pbc_c[i] and pbc_c[j]):
        raise RuntimeError('Axes have to be periodic')
    prod_cc = np.dot(orgcell_cc, orgcell_cc.T)
    cell_cc = 1.0 * orgcell_cc
    nji = np.floor(-prod_cc[i, j] / prod_cc[i, i] + 0.5)
    cell_cc[j] = orgcell_cc[j] + nji * cell_cc[i]

    def volume(cell):
        return np.abs(np.dot(cell[2], np.cross(cell[0], cell[1])))
    V = volume(cell_cc)
    assert abs(volume(orgcell_cc) - V) / V < 1e-10
    atoms.set_cell(cell_cc)
    if fold_atoms:
        atoms.wrap()