from __future__ import annotations
import copy
import itertools
import json
import logging
import math
import os
import warnings
from functools import reduce
from math import gcd, isclose
from typing import TYPE_CHECKING, cast
import numpy as np
from monty.fractions import lcm
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Lattice, PeriodicSite, Structure, get_el_sp
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.util.coord import in_coord_list
from pymatgen.util.due import Doi, due
def get_unreconstructed_slabs(self) -> list[Slab]:
    """Generate the unreconstructed (super) Slabs.

        TODO (@DanielYang59): this should be a private method.
        """
    return [slab.make_supercell(self.trans_matrix) for slab in SlabGenerator(**self.slabgen_params).get_slabs()]