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
def parse_natural_populations(lines: list[str]) -> list[pd.DataFrame]:
    """
    Parse the natural populations section of NBO output.

    Args:
        lines: QChem output lines.

    Returns:
        Data frame of formatted output.

    Raises:
        RuntimeError
    """
    no_failures = True
    pop_dfs = []
    while no_failures:
        try:
            lines = jump_to_header(lines, 'Summary of Natural Population Analysis:')
        except RuntimeError:
            no_failures = False
        if no_failures:
            lines = lines[4:]
            columns = lines[0].split()
            lines = lines[2:]
            data = []
            for line in lines:
                if '=' in line:
                    break
                values = line.split()
                if len(values[0]) > 2:
                    values.insert(0, values[0][0:-3])
                    values[1] = values[1][-3:]
                data.append([str(values[0]), int(values[1]), float(values[2]), float(values[3]), float(values[4]), float(values[5]), float(values[6])])
                if len(columns) == 8:
                    data[-1].append(float(values[7]))
            pop_dfs.append(pd.DataFrame(data=data, columns=columns))
    return pop_dfs