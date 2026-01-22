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
def get_reduced_structure(self, reduction_algo: Literal['niggli', 'LLL']='niggli') -> Self:
    """Get a reduced structure.

        Args:
            reduction_algo ("niggli" | "LLL"): The lattice reduction algorithm to use.
                Defaults to "niggli".

        Returns:
            Structure: Niggli- or LLL-reduced structure.
        """
    if reduction_algo == 'niggli':
        reduced_latt = self._lattice.get_niggli_reduced_lattice()
    elif reduction_algo == 'LLL':
        reduced_latt = self._lattice.get_lll_reduced_lattice()
    else:
        raise ValueError(f"Invalid reduction_algo={reduction_algo!r}, must be 'niggli' or 'LLL'.")
    if reduced_latt != self.lattice:
        return type(self)(reduced_latt, self.species_and_occu, self.cart_coords, coords_are_cartesian=True, to_unit_cell=True, site_properties=self.site_properties, labels=self.labels, charge=self._charge)
    return self.copy()