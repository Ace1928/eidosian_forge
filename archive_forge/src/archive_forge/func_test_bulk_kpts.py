from pytest import approx, fixture
from ase import Atoms
from ase.build import bulk
from ase.vibrations.raman import StaticRamanCalculator
from ase.vibrations.raman import StaticRamanPhononsCalculator
from ase.vibrations.placzek import PlaczekStatic
from ase.vibrations.placzek import PlaczekStaticPhonons
from ase.calculators.bond_polarizability import BondPolarizability
from ase.calculators.emt import EMT
def test_bulk_kpts(Cbulk, testdir):
    """Bulk FCC carbon (for EMT) for phonons"""
    name = 'phbp'
    rm = StaticRamanPhononsCalculator(Cbulk, BondPolarizability, calc=EMT(), name=name, delta=0.05, supercell=(2, 1, 1))
    rm.run()
    pz = PlaczekStaticPhonons(Cbulk, name=name, supercell=(2, 1, 1))
    energies_1kpt = pz.get_energies()
    pz.kpts = (2, 1, 1)
    energies_2kpts = pz.get_energies()
    assert len(energies_2kpts) == 2 * len(energies_1kpt)
    pz.kpts = (2, 1, 2)
    pz.summary()