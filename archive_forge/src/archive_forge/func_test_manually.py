import pytest
from ase.io.formats import ioformats
def test_manually():
    traj = ioformats['traj']
    print(traj)
    outcar = ioformats['vasp-out']
    print(outcar)
    assert outcar.match_name('OUTCAR')
    assert outcar.match_name('something.with.OUTCAR.stuff')