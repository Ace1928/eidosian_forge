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
@classmethod
def from_computed_structure_entry(cls, entry, miller_index, label=None, adsorbates=None, clean_entry=None, **kwargs) -> Self:
    """Returns SlabEntry from a ComputedStructureEntry."""
    return cls(entry.structure, entry.energy, miller_index, label=label, adsorbates=adsorbates, clean_entry=clean_entry, **kwargs)