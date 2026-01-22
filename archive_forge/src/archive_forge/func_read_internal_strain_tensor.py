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
def read_internal_strain_tensor(self):
    """
        Reads the internal strain tensor and populates self.internal_strain_tensor with an array of voigt notation
            tensors for each site.
        """
    search = []

    def internal_strain_start(results, match):
        results.internal_strain_ion = int(match.group(1)) - 1
        results.internal_strain_tensor.append(np.zeros((3, 6)))
    search.append(['INTERNAL STRAIN TENSOR FOR ION\\s+(\\d+)\\s+for displacements in x,y,z  \\(eV/Angst\\):', None, internal_strain_start])

    def internal_strain_data(results, match):
        if match.group(1).lower() == 'x':
            index = 0
        elif match.group(1).lower() == 'y':
            index = 1
        elif match.group(1).lower() == 'z':
            index = 2
        else:
            raise IndexError(f"Couldn't parse row index from symbol for internal strain tensor: {match.group(1)}")
        results.internal_strain_tensor[results.internal_strain_ion][index] = np.array([float(match.group(i)) for i in range(2, 8)])
        if index == 2:
            results.internal_strain_ion = None
    search.append(['^\\s+([x,y,z])\\s+' + '([-]?\\d+\\.\\d+)\\s+' * 6, lambda results, _line: results.internal_strain_ion is not None, internal_strain_data])
    self.internal_strain_ion = None
    self.internal_strain_tensor = []
    micro_pyawk(self.filename, search, self)