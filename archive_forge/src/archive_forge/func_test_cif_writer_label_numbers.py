import io
import numpy as np
import warnings
import pytest
from ase import Atoms
from ase.build import molecule
from ase.io import read, write
from ase.io.cif import CIFLoop, parse_loop, NoStructureData, parse_cif
from ase.calculators.calculator import compare_atoms
def test_cif_writer_label_numbers(cif_atoms):
    cif_atoms.write('testfile.cif')
    atoms1 = read('testfile.cif', store_tags=True)
    labels = atoms1.info['_atom_site_label']
    elements = atoms1.info['_atom_site_type_symbol']
    build_labels = ['{:}{:}'.format(x, i) for x in set(elements) for i in range(1, elements.count(x) + 1)]
    assert build_labels.sort() == labels.sort()