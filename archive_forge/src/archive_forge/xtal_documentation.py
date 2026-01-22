from typing import Dict, Any
import numpy as np
from scipy import spatial
import ase
from ase.symbols import string2symbols
from ase.spacegroup import Spacegroup
from ase.geometry import cellpar_to_cell
Return `sumbols` as a sequence of element symbols.