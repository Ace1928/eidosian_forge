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
@property
def hubbards(self) -> dict[str, float]:
    """Hubbard U values used if a vasprun is a GGA+U run. Otherwise an empty dict."""
    symbols = [s.split()[1] for s in self.potcar_symbols]
    symbols = [re.split('_', s)[0] for s in symbols]
    if not self.incar.get('LDAU', False):
        return {}
    us = self.incar.get('LDAUU', self.parameters.get('LDAUU'))
    js = self.incar.get('LDAUJ', self.parameters.get('LDAUJ'))
    if len(js) != len(us):
        js = [0] * len(us)
    if len(us) == len(symbols):
        return {symbols[idx]: us[idx] - js[idx] for idx in range(len(symbols))}
    if sum(us) == 0 and sum(js) == 0:
        return {}
    raise VaspParseError('Length of U value parameters and atomic symbols are mismatched')