import numpy as np
import pytest
import warnings
from ase import Atom, Atoms
from ase.io import read
from ase.io import NetCDFTrajectory
def test_netcdf_with_nonconsecutive_index(netCDF4):
    nc = netCDF4.Dataset('7.nc', 'w')
    nc.createDimension('frame', None)
    nc.createDimension('atom', 3)
    nc.createDimension('spatial', 3)
    nc.createDimension('cell_spatial', 3)
    nc.createDimension('cell_angular', 3)
    nc.createVariable('atom_types', 'i', ('atom',))
    nc.createVariable('coordinates', 'f4', ('frame', 'atom', 'spatial'))
    nc.createVariable('cell_lengths', 'f4', ('frame', 'cell_spatial'))
    nc.createVariable('cell_angles', 'f4', ('frame', 'cell_angular'))
    nc.createVariable('id', 'i', ('frame', 'atom'))
    r0 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
    r1 = 2 * r0
    nc.variables['atom_types'][:] = [1, 2, 3]
    nc.variables['coordinates'][0] = r0
    nc.variables['coordinates'][1] = r1
    nc.variables['cell_lengths'][:] = 0
    nc.variables['cell_angles'][:] = 90
    nc.variables['id'][0] = [13, 3, 5]
    nc.variables['id'][1] = [-1, 0, -5]
    nc.close()
    traj = NetCDFTrajectory('7.nc', 'r')
    assert (traj[0].numbers == [2, 3, 1]).all()
    assert (traj[1].numbers == [3, 1, 2]).all()
    traj.close()