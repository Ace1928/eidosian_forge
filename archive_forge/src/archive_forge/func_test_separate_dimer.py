from ase import Atoms
from ase.build import molecule
from ase.build.connected import connected_atoms, split_bond, separate
from ase.data.s22 import data
def test_separate_dimer():
    dimerdata = data['Methanol-formaldehyde_complex']
    dimer = Atoms(dimerdata['symbols'], dimerdata['positions'])
    atoms_list = separate(dimer)
    assert len(atoms_list) == 2
    assert len(atoms_list[0]) + len(atoms_list[1]) == len(dimer)
    atoms_list = separate(dimer, scale=1e-05)
    assert len(atoms_list) == len(dimer)