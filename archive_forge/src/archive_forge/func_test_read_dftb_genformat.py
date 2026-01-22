import numpy as np
from io import StringIO
from ase.atoms import Atoms
from ase.units import AUT, Bohr, second
from ase.io.dftb import (read_dftb, read_dftb_lattice,
def test_read_dftb_genformat():
    positions = [[-0.740273308080763, 0.666649653991325, 0.159416494587587], [0.006891486298212, -0.006206095648781, -0.531735097642277], [0.697047663527725, 0.447111938577178, -1.264187748314973], [0.036334158254826, -1.107555496919721, -0.464934648630337]]
    cell = [[3.75, 0.0, 0.0], [1.5, 4.5, 0.0], [0.45, 1.05, 3.75]]
    atoms = Atoms('OCH2', cell=cell, positions=positions)
    atoms.set_pbc(True)
    atoms_new = read_dftb(fd_genformat_periodic)
    assert np.all(atoms_new.numbers == atoms.numbers)
    assert np.allclose(atoms_new.positions, atoms.positions)
    assert np.all(atoms_new.pbc == atoms.pbc)
    assert np.allclose(atoms_new.cell, atoms.cell)
    atoms.set_pbc(False)
    atoms_new = read_dftb(fd_genformat_nonperiodic)
    assert np.all(atoms_new.numbers == atoms.numbers)
    assert np.allclose(atoms_new.positions, atoms.positions)
    assert np.all(atoms_new.pbc == atoms.pbc)
    assert np.allclose(atoms_new.cell, 0.0)