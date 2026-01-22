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
def read_avg_core_poten(self):
    """
        Read the core potential at each ionic step.

        Returns:
            A list for each ionic step containing a list of the average core
            potentials for each atom: [[avg core pot]].

        Example:
            The average core potential of the 2nd atom of the structure at the
            last ionic step is: [-1][1]
        """
    with zopen(self.filename, mode='rt') as foutcar:
        line = foutcar.readline()
        aps = []
        while line != '':
            line = foutcar.readline()
            if 'the norm of the test charge is' in line:
                ap = []
                while line != '':
                    line = foutcar.readline()
                    if 'E-fermi' in line:
                        aps.append(ap)
                        break
                    npots = int((len(line) - 1) / 17)
                    for i in range(npots):
                        start = i * 17
                        ap.append(float(line[start + 8:start + 17]))
    return aps