import os
import pytest
import numpy as np
import ase
import ase.lattice.cubic
from ase.calculators.castep import (Castep, CastepOption,
def parse_posblock(pblock, has_units=False):
    lines = pblock.split('\n')
    units = None
    if has_units:
        units = lines.pop(0).strip()
    pos_lines = []
    while len(lines) > 0:
        l = lines.pop(0).strip()
        if l == '':
            continue
        el, x, y, z = l.split()
        xyz = np.array(list(map(float, [x, y, z])))
        pos_lines.append((el, xyz))
    return (units, pos_lines)