from __future__ import annotations
import datetime
import itertools
import logging
import math
import os
import re
import warnings
import xml.etree.ElementTree as ET
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from glob import glob
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Literal
import numpy as np
from monty.io import reverse_readfile, zopen
from monty.json import MSONable, jsanitize
from monty.os.path import zpath
from monty.re import regrep
from numpy.testing import assert_allclose
from pymatgen.core import Composition, Element, Lattice, Structure
from pymatgen.core.units import unitized
from pymatgen.electronic_structure.bandstructure import (
from pymatgen.electronic_structure.core import Magmom, Orbital, OrbitalType, Spin
from pymatgen.electronic_structure.dos import CompleteDos, Dos
from pymatgen.entries.computed_entries import ComputedEntry, ComputedStructureEntry
from pymatgen.io.common import VolumetricData as BaseVolumetricData
from pymatgen.io.core import ParseError
from pymatgen.io.vasp.inputs import Incar, Kpoints, Poscar, Potcar
from pymatgen.io.wannier90 import Unk
from pymatgen.util.io_utils import clean_lines, micro_pyawk
from pymatgen.util.num import make_symmetric_matrix_from_upper_tri
def write_spin(data_type):
    lines = []
    count = 0
    file.write(f'   {dim[0]}   {dim[1]}   {dim[2]}\n')
    for k, j, i in itertools.product(list(range(dim[2])), list(range(dim[1])), list(range(dim[0]))):
        lines.append(_print_fortran_float(self.data[data_type][i, j, k]))
        count += 1
        if count % 5 == 0:
            file.write(' ' + ''.join(lines) + '\n')
            lines = []
        else:
            lines.append(' ')
    if count % 5 != 0:
        file.write(' ' + ''.join(lines) + ' \n')
    data = self.data_aug.get(data_type, [])
    if isinstance(data, Iterable):
        file.write(''.join(data))