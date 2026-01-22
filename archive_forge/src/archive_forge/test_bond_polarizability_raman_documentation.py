from pytest import approx, fixture
from ase import Atoms
from ase.build import bulk
from ase.vibrations.raman import StaticRamanCalculator
from ase.vibrations.raman import StaticRamanPhononsCalculator
from ase.vibrations.placzek import PlaczekStatic
from ase.vibrations.placzek import PlaczekStaticPhonons
from ase.calculators.bond_polarizability import BondPolarizability
from ase.calculators.emt import EMT
Can we calculate triangular (EMT groundstate) C3?