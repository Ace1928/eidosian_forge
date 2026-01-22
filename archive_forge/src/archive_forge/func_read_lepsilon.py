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
def read_lepsilon(self):
    """
        Reads an LEPSILON run.

        # TODO: Document the actual variables.
        """
    try:
        search = []

        def dielectric_section_start(results, match):
            results.dielectric_index = -1
        search.append(['MACROSCOPIC STATIC DIELECTRIC TENSOR \\(', None, dielectric_section_start])

        def dielectric_section_start2(results, match):
            results.dielectric_index = 0
        search.append(['-------------------------------------', lambda results, _line: results.dielectric_index == -1, dielectric_section_start2])

        def dielectric_data(results, match):
            results.dielectric_tensor[results.dielectric_index, :] = np.array([float(match.group(i)) for i in range(1, 4)])
            results.dielectric_index += 1
        search.append(['^ *([-0-9.Ee+]+) +([-0-9.Ee+]+) +([-0-9.Ee+]+) *$', lambda results, _line: results.dielectric_index >= 0 if results.dielectric_index is not None else None, dielectric_data])

        def dielectric_section_stop(results, match):
            results.dielectric_index = None
        search.append(['-------------------------------------', lambda results, _line: results.dielectric_index >= 1 if results.dielectric_index is not None else None, dielectric_section_stop])
        self.dielectric_index = None
        self.dielectric_tensor = np.zeros((3, 3))

        def piezo_section_start(results, _match):
            results.piezo_index = 0
        search.append(['PIEZOELECTRIC TENSOR  for field in x, y, z        \\(C/m\\^2\\)', None, piezo_section_start])

        def piezo_data(results, match):
            results.piezo_tensor[results.piezo_index, :] = np.array([float(match.group(i)) for i in range(1, 7)])
            results.piezo_index += 1
        search.append(['^ *[xyz] +([-0-9.Ee+]+) +([-0-9.Ee+]+) +([-0-9.Ee+]+) *([-0-9.Ee+]+) +([-0-9.Ee+]+) +([-0-9.Ee+]+)*$', lambda results, _line: results.piezo_index >= 0 if results.piezo_index is not None else None, piezo_data])

        def piezo_section_stop(results, _match):
            results.piezo_index = None
        search.append(['-------------------------------------', lambda results, _line: results.piezo_index >= 1 if results.piezo_index is not None else None, piezo_section_stop])
        self.piezo_index = None
        self.piezo_tensor = np.zeros((3, 6))

        def born_section_start(results, _match):
            results.born_ion = -1
        search.append(['BORN EFFECTIVE CHARGES ', None, born_section_start])

        def born_ion(results, match):
            results.born_ion = int(match.group(1)) - 1
            results.born.append(np.zeros((3, 3)))
        search.append(['ion +([0-9]+)', lambda results, _line: results.born_ion is not None, born_ion])

        def born_data(results, match):
            results.born[results.born_ion][int(match.group(1)) - 1, :] = np.array([float(match.group(i)) for i in range(2, 5)])
        search.append(['^ *([1-3]+) +([-0-9.Ee+]+) +([-0-9.Ee+]+) +([-0-9.Ee+]+)$', lambda results, _line: results.born_ion >= 0 if results.born_ion is not None else results.born_ion, born_data])

        def born_section_stop(results, _match):
            results.born_ion = None
        search.append(['-------------------------------------', lambda results, _line: results.born_ion >= 1 if results.born_ion is not None else results.born_ion, born_section_stop])
        self.born_ion = None
        self.born = []
        micro_pyawk(self.filename, search, self)
        self.born = np.array(self.born)
        self.dielectric_tensor = self.dielectric_tensor.tolist()
        self.piezo_tensor = self.piezo_tensor.tolist()
    except Exception:
        raise RuntimeError('LEPSILON OUTCAR could not be parsed.')