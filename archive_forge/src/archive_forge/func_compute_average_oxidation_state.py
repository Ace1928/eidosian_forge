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
def compute_average_oxidation_state(site):
    """
    Calculates the average oxidation state of a site.

    Args:
        site: Site to compute average oxidation state

    Returns:
        Average oxidation state of site.
    """
    try:
        return sum((sp.oxi_state * occu for sp, occu in site.species.items() if sp is not None))
    except AttributeError:
        pass
    try:
        return site.charge
    except AttributeError:
        raise ValueError('Ewald summation can only be performed on structures that are either oxidation state decorated or have site charges.')