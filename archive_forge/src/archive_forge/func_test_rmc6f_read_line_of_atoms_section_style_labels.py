import numpy as np
from ase import Atoms
from ase.io import read, write
import ase.io.rmc6f as rmc6f
from ase.lattice.compounds import TRI_Fe2O3
def test_rmc6f_read_line_of_atoms_section_style_labels():
    """Test for reading a line of atoms section
    w/ 'labels'-included style for rmc6f
    """
    atom_line = '1 S [1] 0.600452 0.525100 0.442050 1 0 0 0'
    atom_id, props = rmc6f._read_line_of_atoms_section(atom_line.split())
    target_id = 1
    target_props = ['S', 0.600452, 0.5251, 0.44205]
    assert atom_id == target_id
    assert props == target_props