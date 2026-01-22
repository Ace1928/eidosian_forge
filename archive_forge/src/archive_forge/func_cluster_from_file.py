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
@staticmethod
def cluster_from_file(filename):
    """
        Parse the feff input file and return the atomic cluster as a Molecule
        object.

        Args:
            filename (str): path the feff input file

        Returns:
            Molecule: the atomic cluster as Molecule object. The absorbing atom
                is the one at the origin.
        """
    atoms_string = Atoms.atoms_string_from_file(filename)
    lines = [line.split() for line in atoms_string.splitlines()[1:]]
    coords = []
    symbols = []
    for tokens in lines:
        if tokens and (not tokens[0].startswith('*')):
            coords.append([float(val) for val in tokens[:3]])
            symbols.append(tokens[4])
    return Molecule(symbols, coords)