import re
import warnings
from typing import Dict
import numpy as np
import ase  # Annotations
from ase.utils import jsonable
from ase.cell import Cell
def resolve_kpt_path_string(path, special_points):
    paths = parse_path_string(path)
    coords = [np.array([special_points[sym] for sym in subpath]).reshape(-1, 3) for subpath in paths]
    return (paths, coords)