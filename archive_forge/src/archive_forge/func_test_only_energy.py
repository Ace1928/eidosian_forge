import pytest
from ase import Atom, Atoms
from ase.io import Trajectory, read
from ase.constraints import FixBondLength
from ase.calculators.calculator import PropertyNotImplementedError
def test_only_energy():
    with Trajectory('fake.traj', 'w') as t:
        t.write(Atoms('H'), energy=-42.0, forces=[[1, 2, 3]])
    a = read('fake.traj')
    with Trajectory('only-energy.traj', 'w', properties=['energy']) as t:
        t.write(a)
    b = read('only-energy.traj')
    e = b.get_potential_energy()
    assert e + 42 == 0
    with pytest.raises(PropertyNotImplementedError):
        b.get_forces()