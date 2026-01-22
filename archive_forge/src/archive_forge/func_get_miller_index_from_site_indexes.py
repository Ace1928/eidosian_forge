from __future__ import annotations
import collections
import contextlib
import functools
import inspect
import io
import itertools
import json
import math
import os
import random
import re
import sys
import warnings
from abc import ABC, abstractmethod
from fnmatch import fnmatch
from inspect import isclass
from io import StringIO
from typing import TYPE_CHECKING, Any, Callable, Literal, SupportsIndex, cast, get_args
import numpy as np
from monty.dev import deprecated
from monty.io import zopen
from monty.json import MSONable
from numpy import cross, eye
from numpy.linalg import norm
from ruamel.yaml import YAML
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.linalg import expm, polar
from scipy.spatial.distance import squareform
from tabulate import tabulate
from pymatgen.core.bonds import CovalentBond, get_bond_length
from pymatgen.core.composition import Composition
from pymatgen.core.lattice import Lattice, get_points_in_spheres
from pymatgen.core.operations import SymmOp
from pymatgen.core.periodic_table import DummySpecies, Element, Species, get_el_sp
from pymatgen.core.sites import PeriodicSite, Site
from pymatgen.core.units import Length, Mass
from pymatgen.electronic_structure.core import Magmom
from pymatgen.symmetry.maggroups import MagneticSpaceGroup
from pymatgen.util.coord import all_distances, get_angle, lattice_points_in_supercell
def get_miller_index_from_site_indexes(self, site_ids, round_dp=4, verbose=True):
    """Get the Miller index of a plane from a set of sites indexes.

        A minimum of 3 sites are required. If more than 3 sites are given
        the best plane that minimises the distance to all points will be
        calculated.

        Args:
            site_ids (list of int): A list of site indexes to consider. A
                minimum of three site indexes are required. If more than three
                sites are provided, the best plane that minimises the distance
                to all sites will be calculated.
            round_dp (int, optional): The number of decimal places to round the
                miller index to.
            verbose (bool, optional): Whether to print warnings.

        Returns:
            tuple: The Miller index.
        """
    return self.lattice.get_miller_index_from_coords(self.frac_coords[site_ids], coords_are_cartesian=False, round_dp=round_dp, verbose=verbose)