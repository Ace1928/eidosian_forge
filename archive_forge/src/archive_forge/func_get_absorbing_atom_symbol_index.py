from __future__ import annotations
import re
import warnings
from typing import TYPE_CHECKING
import numpy as np
from monty.io import zopen
from monty.json import MSONable
from tabulate import tabulate
from pymatgen.core import Element, Lattice, Molecule, Structure
from pymatgen.io.cif import CifParser
from pymatgen.io.core import ParseError
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.util.io_utils import clean_lines
from pymatgen.util.string import str_delimited
def get_absorbing_atom_symbol_index(absorbing_atom, structure):
    """
    Return the absorbing atom symbol and site index in the given structure.

    Args:
        absorbing_atom (str/int): symbol or site index
        structure (Structure)

    Returns:
        str, int: symbol and site index
    """
    if isinstance(absorbing_atom, str):
        return (absorbing_atom, structure.indices_from_symbol(absorbing_atom)[0])
    if isinstance(absorbing_atom, int):
        return (str(structure[absorbing_atom].specie), absorbing_atom)
    raise ValueError('absorbing_atom must be either specie symbol or site index')