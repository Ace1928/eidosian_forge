from ase import Atoms
from ase.build import molecule
from ase.build.connected import connected_atoms, split_bond, separate
from ase.data.s22 import data
def test_connected_atoms():
    CO = molecule('CO')
    R = CO.get_distance(0, 1)
    assert len(connected_atoms(CO, 0, 1.1 * R)) == 2
    assert len(connected_atoms(CO, 0, 0.9 * R)) == 1
    H2O = molecule('H2O')
    assert len(connected_atoms(H2O, 0)) == 3
    assert len(connected_atoms(H2O, 0, scale=0.9)) == 1
    dimerdata = data['2-pyridoxine_2-aminopyridine_complex']
    dimer = Atoms(dimerdata['symbols'], dimerdata['positions'])
    atoms1 = connected_atoms(dimer, 0)
    atoms2 = connected_atoms(dimer, -1)
    assert len(dimer) == len(atoms1) + len(atoms2)