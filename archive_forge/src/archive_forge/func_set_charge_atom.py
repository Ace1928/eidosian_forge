from __future__ import annotations
import itertools
import re
import warnings
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Literal
import numpy as np
import pandas as pd
from monty.io import zopen
from monty.json import MSONable
from monty.serialization import loadfn
from ruamel.yaml import YAML
from pymatgen.core import Element, Lattice, Molecule, Structure
from pymatgen.core.operations import SymmOp
from pymatgen.util.io_utils import clean_lines
def set_charge_atom(self, charges: dict[int, float]) -> None:
    """
        Set the charges of specific atoms of the data.

        Args:
            charges: A dictionary with atom indexes as keys and
                charges as values, e.g., to set the charge
                of the atom with index 3 to -2, use `{3: -2}`.
        """
    for iat, q in charges.items():
        self.atoms.loc[iat, 'q'] = q