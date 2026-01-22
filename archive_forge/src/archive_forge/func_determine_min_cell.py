from __future__ import annotations
import logging
import math
import warnings
from fractions import Fraction
from itertools import groupby, product
from math import gcd
from string import ascii_lowercase
from typing import TYPE_CHECKING, Callable, Literal
import numpy as np
from joblib import Parallel, delayed
from monty.dev import requires
from monty.fractions import lcm
from monty.json import MSONable
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.analysis.energy_models import SymmetryModel
from pymatgen.analysis.ewald import EwaldSummation
from pymatgen.analysis.gb.grain import GrainBoundaryGenerator
from pymatgen.analysis.local_env import MinimumDistanceNN
from pymatgen.analysis.structure_matcher import SpinComparator, StructureMatcher
from pymatgen.analysis.structure_prediction.substitution_probability import SubstitutionPredictor
from pymatgen.command_line.enumlib_caller import EnumError, EnumlibAdaptor
from pymatgen.command_line.mcsqs_caller import run_mcsqs
from pymatgen.core import DummySpecies, Element, Species, Structure, get_el_sp
from pymatgen.core.surface import SlabGenerator
from pymatgen.electronic_structure.core import Spin
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.icet import IcetSQS
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.transformations.standard_transformations import (
from pymatgen.transformations.transformation_abc import AbstractTransformation
@staticmethod
def determine_min_cell(disordered_structure):
    """Determine the smallest supercell that is able to enumerate
        the provided structure with the given order parameter.
        """

    def lcm(n1, n2):
        """Find least common multiple of two numbers."""
        return n1 * n2 / gcd(n1, n2)
    mag_species_order_parameter = {}
    mag_species_occurrences = {}
    for site in disordered_structure:
        if not site.is_ordered:
            sp = str(next(iter(site.species))).split(',', maxsplit=1)[0]
            if sp in mag_species_order_parameter:
                mag_species_occurrences[sp] += 1
            else:
                op = max(site.species.values())
                mag_species_order_parameter[sp] = op
                mag_species_occurrences[sp] = 1
    smallest_n = []
    for sp, order_parameter in mag_species_order_parameter.items():
        denom = Fraction(order_parameter).limit_denominator(100).denominator
        num_atom_per_specie = mag_species_occurrences[sp]
        n_gcd = gcd(denom, num_atom_per_specie)
        smallest_n.append(lcm(int(n_gcd), denom) / n_gcd)
    return max(smallest_n)