from __future__ import annotations
import bisect
from copy import copy, deepcopy
from datetime import datetime
from math import log, pi, sqrt
from typing import TYPE_CHECKING, Any
from warnings import warn
import numpy as np
from monty.json import MSONable
from scipy import constants
from scipy.special import comb, erfc
from pymatgen.core.structure import Structure
from pymatgen.util.due import Doi, due
def find_match(site):
    for test_site in sub_structure:
        frac_diff = abs(np.array(site.frac_coords) - np.array(test_site.frac_coords)) % 1
        frac_diff = [abs(a) < tol or abs(a) > 1 - tol for a in frac_diff]
        if all(frac_diff):
            return test_site
    return None