from ase.io import read, write
import numpy as np
from ase import Atoms
def test_pdb_cell_io():
    atoms1 = images[0]
    write('grumbles.pdb', atoms1)
    atoms2 = read('grumbles.pdb')
    spos1 = (atoms1.get_scaled_positions() + 0.5) % 1.0
    spos2 = (atoms2.get_scaled_positions() + 0.5) % 1.0
    for a, b in zip(spos1, spos2):
        print(a, b)
    err = np.abs(spos1 - spos2).max()
    print(err)
    assert err < 0.0002