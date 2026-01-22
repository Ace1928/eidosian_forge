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
def z_int(string: str) -> int:
    """
    Convert string to integer.
    If string empty, return -1.

    Args:
        string: Input to be cast to int.

    Returns:
        Int representation.

    Raises:
        n/a
    """
    try:
        return int(string)
    except ValueError:
        return -1