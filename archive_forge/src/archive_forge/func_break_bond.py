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
def break_bond(self, ind1: int, ind2: int, tol: float=0.2) -> tuple[IMolecule | Molecule, ...]:
    """Returns two molecules based on breaking the bond between atoms at index
        ind1 and ind2.

        Args:
            ind1 (int): 1st site index
            ind2 (int): 2nd site index
            tol (float): Relative tolerance to test. Basically, the code
                checks if the distance between the sites is less than (1 +
                tol) * typical bond distances. Defaults to 0.2, i.e.,
                20% longer.

        Returns:
            Two Molecule objects representing the two clusters formed from
            breaking the bond.
        """
    clusters = [[self[ind1]], [self[ind2]]]
    sites = [site for idx, site in enumerate(self) if idx not in (ind1, ind2)]

    def belongs_to_cluster(site, cluster):
        return any((CovalentBond.is_bonded(site, test_site, tol=tol) for test_site in cluster))
    while len(sites) > 0:
        unmatched = []
        for site in sites:
            for cluster in clusters:
                if belongs_to_cluster(site, cluster):
                    cluster.append(site)
                    break
            else:
                unmatched.append(site)
        if len(unmatched) == len(sites):
            raise ValueError('Not all sites are matched!')
        sites = unmatched
    return tuple((type(self).from_sites(cluster) for cluster in clusters))