from ase import Atoms
from ase.calculators.singlepoint import (SinglePointDFTCalculator,
from ase.units import Bohr, Hartree
import ase.io.ulm as ulm
from ase.io.trajectory import read_atoms
Read gpw-file from GPAW.