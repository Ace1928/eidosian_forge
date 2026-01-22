from base64 import b64encode, b64decode
from pathlib import Path
from ase import Atoms
from ase.constraints import FixAtoms
from ase.io import read
from ase.io.trajectory import Trajectory
def test_oldtraj():
    Path('old.traj').write_bytes(b64decode(data))
    a1, a2 = read('old.traj@:')
    assert len(a1.constraints) == 1
    assert len(a2.constraints) == 0
    assert not a1.pbc.any()
    assert a2.pbc.all()