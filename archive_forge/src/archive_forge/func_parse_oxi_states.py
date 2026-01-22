from __future__ import annotations
import math
import os
import re
import textwrap
import warnings
from collections import defaultdict, deque
from functools import partial
from inspect import getfullargspec
from io import StringIO
from itertools import groupby
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
import numpy as np
from monty.dev import deprecated
from monty.io import zopen
from monty.serialization import loadfn
from pymatgen.core import Composition, DummySpecies, Element, Lattice, PeriodicSite, Species, Structure, get_el_sp
from pymatgen.core.operations import MagSymmOp, SymmOp
from pymatgen.electronic_structure.core import Magmom
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer, SpacegroupOperations
from pymatgen.symmetry.groups import SYMM_DATA, SpaceGroup
from pymatgen.symmetry.maggroups import MagneticSpaceGroup
from pymatgen.symmetry.structure import SymmetrizedStructure
from pymatgen.util.coord import find_in_coord_list_pbc, in_coord_list_pbc
@staticmethod
def parse_oxi_states(data):
    """Parse oxidation states from data dictionary."""
    try:
        oxi_states = {data['_atom_type_symbol'][i]: str2float(data['_atom_type_oxidation_number'][i]) for i in range(len(data['_atom_type_symbol']))}
        for i, symbol in enumerate(data['_atom_type_symbol']):
            oxi_states[re.sub('\\d?[\\+,\\-]?$', '', symbol)] = str2float(data['_atom_type_oxidation_number'][i])
    except (ValueError, KeyError):
        oxi_states = None
    return oxi_states