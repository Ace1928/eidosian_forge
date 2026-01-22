import os
import re
import warnings
import numpy as np
from copy import deepcopy
import ase
from ase.parallel import paropen
from ase.spacegroup import Spacegroup
from ase.geometry.cell import cellpar_to_cell
from ase.constraints import FixAtoms, FixedPlane, FixedLine, FixCartesian
from ase.utils import atoms_to_spglib_cell
import ase.units
def parse_blockunit(line_tokens, blockname):
    u = 1.0
    if len(line_tokens[0]) == 1:
        usymb = line_tokens[0][0].lower()
        u = cell_units.get(usymb, 1)
        if usymb not in cell_units:
            warnings.warn('read_cell: Warning - ignoring invalid unit specifier in %BLOCK {0} (assuming Angstrom instead)'.format(blockname))
        line_tokens = line_tokens[1:]
    return (u, line_tokens)