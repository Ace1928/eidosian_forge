import os
import pytest
import numpy as np
from ase.atoms import Atoms
@pytest.mark.calculator_lite
@pytest.mark.calculator('lammpslib')
def test_lammps_neighlist_buf(factory, testdir):
    os.chdir(testdir)
    atoms = Atoms('He', cell=[[2.045, 2.045, 0.0], [2.045, 0.0, 2.045], [0.0, 2.045, 2.045]], pbc=[True] * 3)
    atoms *= 6
    calc = factory.calc(lmpcmds=['pair_style lj/cut 0.5995011000293092E+01', 'pair_coeff * * 3.0 3.0'], atom_types={'H': 1, 'He': 2}, log_file=None, keep_alive=True, lammps_header=['units metal', 'atom_style atomic', 'atom_modify map array sort 0 0'])
    atoms.calc = calc
    f = atoms.get_forces()
    fmag = np.linalg.norm(f, axis=1)
    print(f'> 1e-6 f[{np.where(fmag > 1e-06)}] = {f[np.where(fmag > 1e-06)]}')
    print(f'max f[{np.argmax(fmag)}] = {np.max(fmag)}')
    assert len(np.where(fmag > 1e-10)[0]) == 0