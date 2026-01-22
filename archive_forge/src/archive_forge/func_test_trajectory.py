import pytest
from ase import Atom, Atoms
from ase.io import Trajectory, read
from ase.constraints import FixBondLength
from ase.calculators.calculator import PropertyNotImplementedError
def test_trajectory(trajfile, images):
    imgs = read(trajfile, index=':')
    assert len(imgs) == len(images)
    for img1, img2 in zip(imgs, images):
        assert img1 == img2
    with Trajectory(trajfile, 'r') as read_traj:
        sliced_traj = read_traj[3:8]
        assert len(sliced_traj) == 5
        sliced_again = sliced_traj[1:-1]
        assert len(sliced_again) == 3
        assert sliced_traj[1] == sliced_again[0]