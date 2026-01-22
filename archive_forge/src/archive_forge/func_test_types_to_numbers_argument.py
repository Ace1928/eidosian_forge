import numpy as np
import pytest
import warnings
from ase import Atom, Atoms
from ase.io import read
from ase.io import NetCDFTrajectory
def test_types_to_numbers_argument(co):
    traj = NetCDFTrajectory('8.nc', 'w', co)
    traj.write()
    traj.close()
    d = {6: 15, 8: 15}
    traj = NetCDFTrajectory('8.nc', mode='r', types_to_numbers=d)
    assert np.allclose(traj[-1].get_masses(), 30.974)
    assert (traj[-1].numbers == [15, 15]).all()
    d = {3: 14}
    traj = NetCDFTrajectory('8.nc', mode='r', types_to_numbers=d)
    assert (traj[-1].numbers == [6, 8]).all()
    traj = NetCDFTrajectory('8.nc', 'r', types_to_numbers=[0, 0, 0, 0, 0, 0, 15])
    assert (traj[-1].numbers == [15, 8]).all()
    traj.close()