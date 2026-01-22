from numpy.testing import assert_allclose
import ase.build
from ase.calculators.emt import EMT
def test_potential_energies():
    TOL = 1e-08
    atoms = ase.build.bulk('Ni', crystalstructure='fcc', cubic=1)
    atoms *= (2, 2, 2)
    atoms.calc = EMT()
    energies = atoms.get_potential_energies()
    energy = atoms.get_potential_energy()
    assert abs(energies.sum() - energy) < TOL
    assert_allclose(energies, energies[0], rtol=TOL)
    atoms.rattle()
    assert abs(energies.sum() - energy) < TOL