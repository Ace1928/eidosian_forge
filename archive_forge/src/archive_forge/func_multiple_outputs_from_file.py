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
@staticmethod
def multiple_outputs_from_file(filename, keep_sub_files=True):
    """
        Parses a QChem output file with multiple calculations
        # 1.) Separates the output into sub-files
            e.g. qcout -> qcout.0, qcout.1, qcout.2 ... qcout.N
            a.) Find delimiter for multiple calculations
            b.) Make separate output sub-files
        2.) Creates separate QCCalcs for each one from the sub-files.
        """
    to_return = []
    with zopen(filename, mode='rt') as file:
        text = re.split('\\s*(?:Running\\s+)*Job\\s+\\d+\\s+of\\s+\\d+\\s+', file.read())
    if text[0] == '':
        text = text[1:]
    for i, sub_text in enumerate(text):
        with open(f'{filename}.{i}', mode='w') as temp:
            temp.write(sub_text)
        tempOutput = QCOutput(f'{filename}.{i}')
        to_return.append(tempOutput)
        if not keep_sub_files:
            os.remove(f'{filename}.{i}')
    return to_return