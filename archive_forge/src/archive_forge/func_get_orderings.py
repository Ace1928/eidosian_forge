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
def get_orderings(self, mode: Literal['enum', 'sqs']='enum', **kwargs) -> list[Structure]:
    """Returns list of orderings for a disordered structure. If structure
        does not contain disorder, the default structure is returned.

        Args:
            mode ("enum" | "sqs"): Either "enum" or "sqs". If enum,
                the enumlib will be used to return all distinct
                orderings. If sqs, mcsqs will be used to return
                an sqs structure.
            kwargs: kwargs passed to either
                pymatgen.command_line..enumlib_caller.EnumlibAdaptor
                or pymatgen.command_line.mcsqs_caller.run_mcsqs.
                For run_mcsqs, a default cluster search of 2 cluster interactions
                with 1NN distance and 3 cluster interactions with 2NN distance
                is set.

        Returns:
            List[Structure]
        """
    if self.is_ordered:
        return [self]
    if mode.startswith('enum'):
        from pymatgen.command_line.enumlib_caller import EnumlibAdaptor
        adaptor = EnumlibAdaptor(self, **kwargs)
        adaptor.run()
        return adaptor.structures
    if mode == 'sqs':
        from pymatgen.command_line.mcsqs_caller import run_mcsqs
        if 'clusters' not in kwargs:
            disordered_sites = [site for site in self if not site.is_ordered]
            subset_structure = Structure.from_sites(disordered_sites)
            dist_matrix = subset_structure.distance_matrix
            dists = sorted(set(dist_matrix.ravel()))
            unique_dists = []
            for idx in range(1, len(dists)):
                if dists[idx] - dists[idx - 1] > 0.1:
                    unique_dists.append(dists[idx])
            clusters = {idx + 2: dist + 0.01 for idx, dist in enumerate(unique_dists) if idx < 2}
            kwargs['clusters'] = clusters
        return [run_mcsqs(self, **kwargs).bestsqs]
    raise ValueError('Invalid mode!')