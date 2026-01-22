from pytest import approx, fixture
from ase import Atoms
from ase.build import bulk
from ase.vibrations.raman import StaticRamanCalculator
from ase.vibrations.raman import StaticRamanPhononsCalculator
from ase.vibrations.placzek import PlaczekStatic
from ase.vibrations.placzek import PlaczekStaticPhonons
from ase.calculators.bond_polarizability import BondPolarizability
from ase.calculators.emt import EMT
def test_c3(testdir):
    """Can we calculate triangular (EMT groundstate) C3?"""
    y, z = (0.30646191, 1.14411339)
    atoms = Atoms('C3', positions=[[0, 0, 0], [0, y, z], [0, z, y]])
    atoms.calc = EMT()
    name = 'bp'
    rm = StaticRamanCalculator(atoms, BondPolarizability, name=name, exname=name, txt='-')
    rm.run()
    pz = PlaczekStatic(atoms, name=name)
    i_vib = pz.get_absolute_intensities()
    assert i_vib[-3:] == approx([5.36301901, 5.36680555, 35.7323934], 1e-06)
    pz.summary()