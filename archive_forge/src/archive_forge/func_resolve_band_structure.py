import os
import re
import tempfile
import warnings
import shutil
from os.path import join, isfile, islink
import numpy as np
from ase.units import Ry, eV, Bohr
from ase.data import atomic_numbers
from ase.io.siesta import read_siesta_xv
from ase.calculators.siesta.import_functions import read_rho
from ase.calculators.siesta.import_functions import \
from ase.calculators.calculator import FileIOCalculator, ReadError
from ase.calculators.calculator import Parameters, all_changes
from ase.calculators.siesta.parameters import PAOBasisBlock, Species
from ase.calculators.siesta.parameters import format_fdf
def resolve_band_structure(path, kpts, energies, efermi):
    """Convert input BandPath along with Siesta outputs into BS object."""
    from ase.spectrum.band_structure import BandStructure
    ksn2e = energies
    skn2e = np.swapaxes(ksn2e, 0, 1)
    bs = BandStructure(path, skn2e, reference=efermi)
    return bs