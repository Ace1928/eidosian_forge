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
def species(self, atoms):
    """Find all relevant species depending on the atoms object and
        species input.

            Parameters :
                - atoms : An Atoms object.
        """
    symbols = np.array(atoms.get_chemical_symbols())
    tags = atoms.get_tags()
    species = list(self['species'])
    default_species = [s for s in species if s['tag'] is None and s['symbol'] in symbols]
    default_symbols = [s['symbol'] for s in default_species]
    for symbol in symbols:
        if symbol not in default_symbols:
            spec = Species(symbol=symbol, basis_set=self['basis_set'], tag=None)
            default_species.append(spec)
            default_symbols.append(symbol)
    assert len(default_species) == len(np.unique(symbols))
    species_numbers = np.zeros(len(atoms), int)
    i = 1
    for spec in default_species:
        mask = symbols == spec['symbol']
        species_numbers[mask] = i
        i += 1
    non_default_species = [s for s in species if not s['tag'] is None]
    for spec in non_default_species:
        mask1 = tags == spec['tag']
        mask2 = symbols == spec['symbol']
        mask = np.logical_and(mask1, mask2)
        if sum(mask) > 0:
            species_numbers[mask] = i
            i += 1
    all_species = default_species + non_default_species
    return (all_species, species_numbers)