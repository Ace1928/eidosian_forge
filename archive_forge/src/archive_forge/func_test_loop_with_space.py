import io
import numpy as np
import warnings
import pytest
from ase import Atoms
from ase.build import molecule
from ase.io import read, write
from ase.io.cif import CIFLoop, parse_loop, NoStructureData, parse_cif
from ase.calculators.calculator import compare_atoms
def test_loop_with_space():
    buf = io.BytesIO(cif_with_whitespace_after_loop)
    blocks = list(parse_cif(buf))
    assert len(blocks) == 1
    assert blocks[0]['_potato'] == 42