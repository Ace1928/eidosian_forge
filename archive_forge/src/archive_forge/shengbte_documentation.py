from __future__ import annotations
import warnings
from typing import TYPE_CHECKING, Any
import numpy as np
from monty.dev import requires
from monty.json import MSONable
from pymatgen.core.structure import Structure
from pymatgen.io.vasp import Kpoints
Returns: MSONable dict.