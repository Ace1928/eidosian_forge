import io
import numpy as np
import warnings
import pytest
from ase import Atoms
from ase.build import molecule
from ase.io import read, write
from ase.io.cif import CIFLoop, parse_loop, NoStructureData, parse_cif
from ase.calculators.calculator import compare_atoms
def check_fractional_occupancies(atoms):
    """ Checks fractional occupancy entries in atoms.info dict """
    assert atoms.info['occupancy']
    assert list(atoms.arrays['spacegroup_kinds'])
    occupancies = atoms.info['occupancy']
    for key in occupancies:
        assert isinstance(key, str)
    kinds = atoms.arrays['spacegroup_kinds']
    for a in atoms:
        a_index_str = str(kinds[a.index])
        if a.symbol == 'Na':
            assert len(occupancies[a_index_str]) == 2
            assert occupancies[a_index_str]['K'] == 0.25
            assert occupancies[a_index_str]['Na'] == 0.75
        else:
            assert len(occupancies[a_index_str]) == 1
        if a.symbol == 'Cl':
            assert occupancies[a_index_str]['Cl'] == 0.3