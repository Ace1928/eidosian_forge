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
def replace_species(self, species_mapping: dict[SpeciesLike, SpeciesLike | dict[SpeciesLike, float]], in_place: bool=True) -> SiteCollection:
    """Swap species.

        Note that this clears the label of any affected site.

        Args:
            species_mapping (dict): Species to swap. Species can be elements too. E.g.,
                {Element("Li"): Element("Na")} performs a Li for Na substitution. The second species can
                be a sp_and_occu dict. For example, a site with 0.5 Si that is passed the mapping
                {Element('Si'): {Element('Ge'): 0.75, Element('C'): 0.25} } will have .375 Ge and .125 C.
            in_place (bool): Whether to perform the substitution in place or modify a copy.
                Defaults to True.

        Returns:
            SiteCollection: self or new SiteCollection (depending on in_place) with species replaced.
        """
    site_coll = self if in_place else self.copy()
    sp_mapping = {get_el_sp(k): v for k, v in species_mapping.items()}
    sp_to_replace = set(sp_mapping)
    sp_in_structure = set(self.composition)
    if not sp_in_structure >= sp_to_replace:
        warnings.warn(f'Some species to be substituted are not present in structure. Pls check your input. Species to be substituted = {sp_to_replace}; Species in structure = {sp_in_structure}')
    for site in site_coll:
        if sp_to_replace.intersection(site.species):
            comp = Composition()
            for sp, amt in site.species.items():
                new_sp = sp_mapping.get(sp, sp)
                try:
                    comp += Composition(new_sp) * amt
                except Exception:
                    comp += {new_sp: amt}
            site.species = comp
            site.label = None
    return site_coll