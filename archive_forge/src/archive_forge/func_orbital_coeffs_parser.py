from __future__ import annotations
import copy
import logging
import math
import os
import re
import struct
import warnings
from typing import TYPE_CHECKING, Any
import networkx as nx
import numpy as np
import pandas as pd
from monty.io import zopen
from monty.json import MSONable, jsanitize
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.analysis.local_env import OpenBabelNN
from pymatgen.core import Molecule
from pymatgen.io.qchem.utils import (
def orbital_coeffs_parser(filename: str='53.0') -> NDArray:
    """
    Parse the orbital coefficients from a scratch file.

    Args:
        filename: Path to the orbital coefficients file. Defaults to "53.0".

    Returns:
        NDArray: The orbital coefficients
    """
    orbital_coeffs: list[float] = []
    with zopen(filename, mode='rb') as file:
        binary = file.read()
    orbital_coeffs.extend((struct.unpack('d', binary[ii * 8:(ii + 1) * 8])[0] for ii in range(len(binary) // 8)))
    return np.array(orbital_coeffs)