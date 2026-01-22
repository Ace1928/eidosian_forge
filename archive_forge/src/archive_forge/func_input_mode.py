from __future__ import annotations
import re
from collections import defaultdict
from typing import TYPE_CHECKING
from monty.io import zopen
from monty.re import regrep
from pymatgen.core import Element, Lattice, Structure
from pymatgen.util.io_utils import clean_lines
def input_mode(line):
    if line[0] == '&':
        return ('sections', line[1:].lower())
    if 'ATOMIC_SPECIES' in line:
        return ('pseudo',)
    if 'K_POINTS' in line:
        return ('kpoints', line.split()[1])
    if 'OCCUPATIONS' in line:
        return 'occupations'
    if 'CELL_PARAMETERS' in line or 'ATOMIC_POSITIONS' in line:
        return ('structure', line.split()[1])
    if line == '/':
        return None
    return mode