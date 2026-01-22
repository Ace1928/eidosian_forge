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
def get_computed_entry(self, inc_structure=True, parameters=None, data=None, entry_id: str | None=None):
    """
        Returns a ComputedEntry or ComputedStructureEntry from the Vasprun.

        Args:
            inc_structure (bool): Set to True if you want
                ComputedStructureEntries to be returned instead of
                ComputedEntries.
            parameters (list): Input parameters to include. It has to be one of
                the properties supported by the Vasprun object. If
                parameters is None, a default set of parameters that are
                necessary for typical post-processing will be set.
            data (list): Output data to include. Has to be one of the properties
                supported by the Vasprun object.
            entry_id (str): Specify an entry id for the ComputedEntry. Defaults to
                "vasprun-{current datetime}"

        Returns:
            ComputedStructureEntry/ComputedEntry
        """
    if entry_id is None:
        entry_id = f'vasprun-{datetime.datetime.now()}'
    param_names = {'is_hubbard', 'hubbards', 'potcar_symbols', 'potcar_spec', 'run_type'}
    if parameters:
        param_names.update(parameters)
    params = {p: getattr(self, p) for p in param_names}
    data = {p: getattr(self, p) for p in data} if data is not None else {}
    if inc_structure:
        return ComputedStructureEntry(self.final_structure, self.final_energy, parameters=params, data=data, entry_id=entry_id)
    return ComputedEntry(self.final_structure.composition, self.final_energy, parameters=params, data=data, entry_id=entry_id)