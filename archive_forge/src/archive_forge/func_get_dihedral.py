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
def get_dihedral(self, i: int, j: int, k: int, l: int) -> float:
    """Returns dihedral angle specified by four sites.

        Args:
            i (int): 1st site index
            j (int): 2nd site index
            k (int): 3rd site index
            l (int): 4th site index

        Returns:
            Dihedral angle in degrees.
        """
    vec1 = self[k].coords - self[l].coords
    vec2 = self[j].coords - self[k].coords
    vec3 = self[i].coords - self[j].coords
    vec23 = np.cross(vec2, vec3)
    vec12 = np.cross(vec1, vec2)
    return math.degrees(math.atan2(np.linalg.norm(vec2) * np.dot(vec1, vec23), np.dot(vec12, vec23)))