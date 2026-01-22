from __future__ import annotations
import gzip
import json
from glob import glob
from pathlib import Path
from typing import Any
import numpy as np
from monty.io import zopen
from pymatgen.core import Molecule, Structure
def check_band(test_line: str, ref_line: str) -> bool:
    """Check if band lines are the same.

    Args:
        test_line (str): Line generated in the test file
        ref_line (str): Line generated for the reference file

    Returns:
    bool: True if all points in the test and ref lines are the same
    """
    test_pts = [float(inp) for inp in test_line.split()[-9:-2]]
    ref_pts = [float(inp) for inp in ref_line.split()[-9:-2]]
    return np.allclose(test_pts, ref_pts) and test_line.split()[-2:] == ref_line.split()[-2:]