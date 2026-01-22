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
def extract_cluster(self, target_sites: list[Site], **kwargs) -> list[Site]:
    """Extracts a cluster of atoms based on bond lengths.

        Args:
            target_sites (list[Site]): Initial sites from which to nucleate cluster.
            **kwargs: kwargs passed through to CovalentBond.is_bonded.

        Returns:
            list[Site/PeriodicSite] Cluster of atoms.
        """
    cluster = list(target_sites)
    others = [site for site in self if site not in cluster]
    size = 0
    while len(cluster) > size:
        size = len(cluster)
        new_others = []
        for site in others:
            for site2 in cluster:
                if CovalentBond.is_bonded(site, site2, **kwargs):
                    cluster.append(site)
                    break
            else:
                new_others.append(site)
        others = new_others
    return cluster