import io
import numpy as np
from ase import Atoms
from ase.units import Ha, Bohr
from ase.calculators.openmx.reader import read_openmx, read_eigenvalues
def test_openmx_out():
    with open('openmx_fio_test.out', 'w') as fd:
        fd.write(openmx_out_sample)
    atoms = read_openmx('openmx_fio_test')
    tol = 0.01
    energy = -8.0551
    energies = np.array([-6.2612, -0.4459, -0.4459, -0.4509, -0.4509])
    forces = np.array([[0.0, 0.0, -0.091659], [0.027, 0.027, 0.029454], [-0.027, -0.027, 0.029455], [0.00894, -0.00894, 0.016362], [-0.00894, 0.00894, 0.016362]])
    assert isinstance(atoms, Atoms)
    assert np.isclose(atoms.calc.results['energy'], energy * Ha, atol=tol)
    assert np.all(np.isclose(atoms.calc.results['energies'], energies * Ha, atol=tol))
    assert np.all(np.isclose(atoms.calc.results['forces'], forces * Ha / Bohr, atol=tol))