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
def get_surface_equilibrium(self, slab_entries, delu_dict=None):
    """
        Takes in a list of SlabEntries and calculates the chemical potentials
            at which all slabs in the list coexists simultaneously. Useful for
            building surface phase diagrams. Note that to solve for x equations
            (x slab_entries), there must be x free variables (chemical potentials).
            Adjust delu_dict as need be to get the correct number of free variables.

        Args:
            slab_entries (array): The coefficients of the first equation
            delu_dict (dict): Dictionary of the chemical potentials to be set as
                constant. Note the key should be a sympy Symbol object of the
                format: Symbol("delu_el") where el is the name of the element.

        Returns:
            array: Array containing a solution to x equations with x
                variables (x-1 chemical potential and 1 surface energy)
        """
    all_parameters = []
    all_eqns = []
    for slab_entry in slab_entries:
        se = self.surfe_dict[slab_entry]
        if type(se).__name__ == 'float':
            all_eqns.append(se - Symbol('gamma'))
        else:
            se = sub_chempots(se, delu_dict) if delu_dict else se
            all_eqns.append(se - Symbol('gamma'))
            all_parameters.extend([p for p in list(se.free_symbols) if p not in all_parameters])
    all_parameters.append(Symbol('gamma'))
    solution = linsolve(all_eqns, all_parameters)
    if not solution:
        warnings.warn('No solution')
        return solution
    return {param: next(iter(solution))[idx] for idx, param in enumerate(all_parameters)}