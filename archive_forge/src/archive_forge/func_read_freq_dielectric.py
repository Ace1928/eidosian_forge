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
def read_freq_dielectric(self):
    """
        Parses the frequency dependent dielectric function (obtained with
        LOPTICS). Frequencies (in eV) are in self.frequencies, and dielectric
        tensor function is given as self.dielectric_tensor_function.
        """
    plasma_pattern = 'plasma frequency squared.*'
    dielectric_pattern = 'frequency dependent\\s+IMAGINARY DIELECTRIC FUNCTION \\(independent particle, no local field effects\\)(\\sdensity-density)*$'
    row_pattern = '\\s+'.join(['([\\.\\-\\d]+)'] * 3)
    plasma_frequencies = defaultdict(list)
    read_plasma = False
    read_dielectric = False
    energies = []
    data = {'REAL': [], 'IMAGINARY': []}
    count = 0
    component = 'IMAGINARY'
    with zopen(self.filename, mode='rt') as file:
        for line in file:
            line = line.strip()
            if re.match(plasma_pattern, line):
                read_plasma = 'intraband' if 'intraband' in line else 'interband'
            elif re.match(dielectric_pattern, line):
                read_plasma = False
                read_dielectric = True
                row_pattern = '\\s+'.join(['([\\.\\-\\d]+)'] * 7)
            if read_plasma and re.match(row_pattern, line):
                plasma_frequencies[read_plasma].append([float(t) for t in line.strip().split()])
            elif read_plasma and Outcar._parse_sci_notation(line):
                plasma_frequencies[read_plasma].append(Outcar._parse_sci_notation(line))
            elif read_dielectric:
                tokens = None
                if re.match(row_pattern, line.strip()):
                    tokens = line.strip().split()
                elif Outcar._parse_sci_notation(line.strip()):
                    tokens = Outcar._parse_sci_notation(line.strip())
                elif re.match('\\s*-+\\s*', line):
                    count += 1
                if tokens:
                    if component == 'IMAGINARY':
                        energies.append(float(tokens[0]))
                    xx, yy, zz, xy, yz, xz = (float(t) for t in tokens[1:])
                    matrix = [[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]]
                    data[component].append(matrix)
                if count == 2:
                    component = 'REAL'
                elif count == 3:
                    break
    self.plasma_frequencies = {k: np.array(v[:3]) for k, v in plasma_frequencies.items()}
    self.dielectric_energies = np.array(energies)
    self.dielectric_tensor_function = np.array(data['REAL']) + 1j * np.array(data['IMAGINARY'])