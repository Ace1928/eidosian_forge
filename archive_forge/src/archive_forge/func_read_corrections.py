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
def read_corrections(self, reverse=True, terminate_on_match=True):
    """
        Reads the dipol qudropol corrections into the
        Outcar.data["dipol_quadrupol_correction"].

        Args:
            reverse (bool): Whether to start from end of OUTCAR. Defaults to True.
            terminate_on_match (bool): Whether to terminate once match is found. Defaults to True.
        """
    patterns = {'dipol_quadrupol_correction': 'dipol\\+quadrupol energy correction\\s+([\\d\\-\\.]+)'}
    self.read_pattern(patterns, reverse=reverse, terminate_on_match=terminate_on_match, postprocess=float)
    self.data['dipol_quadrupol_correction'] = self.data['dipol_quadrupol_correction'][0][0]