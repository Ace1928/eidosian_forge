from __future__ import annotations
import copy
import itertools
import random
import warnings
from typing import TYPE_CHECKING
import matplotlib.pyplot as plt
import numpy as np
from sympy import Symbol
from sympy.solvers import linsolve, solve
from pymatgen.analysis.wulff import WulffShape
from pymatgen.core import Structure
from pymatgen.core.composition import Composition
from pymatgen.core.surface import get_slab_regions
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.io.vasp.outputs import Locpot, Outcar
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.util.due import Doi, due
from pymatgen.util.plotting import pretty_plot
def entry_dict_from_list(all_slab_entries):
    """
    Converts a list of SlabEntry to an appropriate dictionary. It is
    assumed that if there is no adsorbate, then it is a clean SlabEntry
    and that adsorbed SlabEntry has the clean_entry parameter set.

    Args:
        all_slab_entries (list): List of SlabEntry objects

    Returns:
        dict: Dictionary of SlabEntry with the Miller index as the main
            key to a dictionary with a clean SlabEntry as the key to a
            list of adsorbed SlabEntry.
    """
    entry_dict = {}
    for entry in all_slab_entries:
        hkl = tuple(entry.miller_index)
        if hkl not in entry_dict:
            entry_dict[hkl] = {}
        clean = entry.clean_entry or entry
        if clean not in entry_dict[hkl]:
            entry_dict[hkl][clean] = []
        if entry.adsorbates:
            entry_dict[hkl][clean].append(entry)
    return entry_dict