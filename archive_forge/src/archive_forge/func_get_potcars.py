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
def get_potcars(self, path: str | Path | bool) -> Potcar | None:
    """Returns the POTCAR from the specified path.

        Args:
            path (str | Path | bool): If a str or Path, the path to search for POTCARs.
                If a bool, whether to take the search path from the specified vasprun.xml

        Returns:
            Potcar | None: The POTCAR from the specified path or None if not found/no path specified.
        """
    if not path:
        return None
    if isinstance(path, (str, Path)) and 'POTCAR' in str(path):
        potcar_paths = [str(path)]
    else:
        search_path = os.path.dirname(os.path.abspath(self.filename)) if path is True else str(path)
        potcar_paths = [f'{search_path}/{fn}' for fn in os.listdir(search_path) if fn.startswith('POTCAR') and '.spec' not in fn]
    for potcar_path in potcar_paths:
        try:
            potcar = Potcar.from_file(potcar_path)
            if {d.header for d in potcar} == set(self.potcar_symbols):
                return potcar
        except Exception:
            continue
    warnings.warn('No POTCAR file with matching TITEL fields was found in\n' + '\n  '.join(potcar_paths))
    return None