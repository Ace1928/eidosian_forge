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
def from_spacegroup(cls, sg: str | int, lattice: list | np.ndarray | Lattice, species: Sequence[str | Element | Species | DummySpecies | Composition], coords: Sequence[Sequence[float]], site_properties: dict[str, Sequence] | None=None, coords_are_cartesian: bool=False, tol: float=1e-05, labels: Sequence[str | None] | None=None) -> IStructure | Structure:
    """Generate a structure using a spacegroup. Note that only symmetrically
        distinct species and coords should be provided. All equivalent sites
        are generated from the spacegroup operations.

        Args:
            sg (str/int): The spacegroup. If a string, it will be interpreted
                as one of the notations supported by
                pymatgen.symmetry.groups.Spacegroup. E.g., "R-3c" or "Fm-3m".
                If an int, it will be interpreted as an international number.
            lattice (Lattice/3x3 array): The lattice, either as a
                pymatgen.core.Lattice or
                simply as any 2D array. Each row should correspond to a lattice
                vector. E.g., [[10,0,0], [20,10,0], [0,0,30]] specifies a
                lattice with lattice vectors [10,0,0], [20,10,0] and [0,0,30].
                Note that no attempt is made to check that the lattice is
                compatible with the spacegroup specified. This may be
                introduced in a future version.
            species ([Species]): Sequence of species on each site. Can take in
                flexible input, including:

                i.  A sequence of element / species specified either as string
                    symbols, e.g. ["Li", "Fe2+", "P", ...] or atomic numbers,
                    e.g., (3, 56, ...) or actual Element or Species objects.

                ii. List of dict of elements/species and occupancies, e.g.,
                    [{"Fe" : 0.5, "Mn":0.5}, ...]. This allows the setup of
                    disordered structures.
            coords (Nx3 array): list of fractional/cartesian coordinates of
                each species.
            coords_are_cartesian (bool): Set to True if you are providing
                coordinates in Cartesian coordinates. Defaults to False.
            site_properties (dict): Properties associated with the sites as a
                dict of sequences, e.g., {"magmom":[5,5,5,5]}. The sequences
                have to be the same length as the atomic species and
                fractional_coords. Defaults to None for no properties.
            tol (float): A fractional tolerance to deal with numerical
               precision issues in determining if orbits are the same.
            labels (list[str]): Labels associated with the sites as a
                list of strings, e.g. ['Li1', 'Li2']. Must have the same
                length as the species and fractional coords. Defaults to
                None for no labels.
        """
    from pymatgen.symmetry.groups import SpaceGroup
    try:
        num = int(sg)
        spg = SpaceGroup.from_int_number(num)
    except ValueError:
        spg = SpaceGroup(sg)
    lattice = lattice if isinstance(lattice, Lattice) else Lattice(lattice)
    if not spg.is_compatible(lattice):
        raise ValueError(f'Supplied lattice with parameters {lattice.parameters} is incompatible with supplied spacegroup {spg.symbol}!')
    if len(species) != len(coords):
        raise ValueError(f'Supplied species and coords lengths ({len(species)} vs {len(coords)}) are different!')
    frac_coords = lattice.get_fractional_coords(coords) if coords_are_cartesian else np.array(coords, dtype=np.float64)
    props = {} if site_properties is None else site_properties
    all_sp: list[str | Element | Species | DummySpecies | Composition] = []
    all_coords: list[list[float]] = []
    all_site_properties: dict[str, list] = collections.defaultdict(list)
    all_labels: list[str | None] = []
    for idx, (sp, c) in enumerate(zip(species, frac_coords)):
        cc = spg.get_orbit(c, tol=tol)
        all_sp.extend([sp] * len(cc))
        all_coords.extend(cc)
        label = labels[idx] if labels else None
        all_labels.extend([label] * len(cc))
        for k, v in props.items():
            all_site_properties[k].extend([v[idx]] * len(cc))
    return cls(lattice, all_sp, all_coords, site_properties=all_site_properties, labels=all_labels)