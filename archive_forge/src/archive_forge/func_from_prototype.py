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
@classmethod
def from_prototype(cls, prototype: str, species: Sequence, **kwargs) -> Self:
    """Method to rapidly construct common prototype structures.

        Args:
            prototype: Name of prototype. E.g., cubic, rocksalt, perovksite etc.
            species: List of species corresponding to symmetrically distinct sites.
            **kwargs: Lattice parameters, e.g., a = 3.0, b = 4, c = 5. Only the required lattice parameters need to be
                specified. For example, if it is a cubic prototype, only a needs to be specified.

        Returns:
            Structure: with given prototype and species.
        """
    prototype = prototype.lower()
    try:
        if prototype == 'fcc':
            return Structure.from_spacegroup('Fm-3m', Lattice.cubic(kwargs['a']), species, [[0, 0, 0]])
        if prototype == 'bcc':
            return Structure.from_spacegroup('Im-3m', Lattice.cubic(kwargs['a']), species, [[0, 0, 0]])
        if prototype == 'hcp':
            return Structure.from_spacegroup('P6_3/mmc', Lattice.hexagonal(kwargs['a'], kwargs['c']), species, [[1 / 3, 2 / 3, 1 / 4]])
        if prototype == 'diamond':
            return Structure.from_spacegroup('Fd-3m', Lattice.cubic(kwargs['a']), species, [[0, 0, 0]])
        if prototype == 'rocksalt':
            return Structure.from_spacegroup('Fm-3m', Lattice.cubic(kwargs['a']), species, [[0, 0, 0], [0.5, 0.5, 0]])
        if prototype == 'perovskite':
            return Structure.from_spacegroup('Pm-3m', Lattice.cubic(kwargs['a']), species, [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0]])
        if prototype == 'cscl':
            return Structure.from_spacegroup('Pm-3m', Lattice.cubic(kwargs['a']), species, [[0, 0, 0], [0.5, 0.5, 0.5]])
        if prototype in {'fluorite', 'caf2'}:
            return Structure.from_spacegroup('Fm-3m', Lattice.cubic(kwargs['a']), species, [[0, 0, 0], [1 / 4, 1 / 4, 1 / 4]])
        if prototype == 'antifluorite':
            return Structure.from_spacegroup('Fm-3m', Lattice.cubic(kwargs['a']), species, [[1 / 4, 1 / 4, 1 / 4], [0, 0, 0]])
        if prototype == 'zincblende':
            return Structure.from_spacegroup('F-43m', Lattice.cubic(kwargs['a']), species, [[0, 0, 0], [1 / 4, 1 / 4, 3 / 4]])
    except KeyError as exc:
        raise ValueError(f'Required parameter {exc} not specified as a kwargs!') from exc
    raise ValueError(f'Unsupported prototype={prototype!r}!')