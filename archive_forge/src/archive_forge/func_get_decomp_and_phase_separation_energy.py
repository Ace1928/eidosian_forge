from __future__ import annotations
import collections
import itertools
import json
import logging
import math
import os
import re
import warnings
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Literal, no_type_check
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from matplotlib import cm
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.font_manager import FontProperties
from monty.json import MontyDecoder, MSONable
from scipy import interpolate
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from tqdm import tqdm
from pymatgen.analysis.reaction_calculator import Reaction, ReactionError
from pymatgen.core import DummySpecies, Element, get_el_sp
from pymatgen.core.composition import Composition
from pymatgen.entries import Entry
from pymatgen.util.coord import Simplex, in_coord_list
from pymatgen.util.due import Doi, due
from pymatgen.util.plotting import pretty_plot
from pymatgen.util.string import htmlify, latexify
def get_decomp_and_phase_separation_energy(self, entry: PDEntry, space_limit: int=200, stable_only: bool=False, tols: Sequence[float]=(1e-08,), maxiter: int=1000, **kwargs: Any) -> tuple[dict[PDEntry, float], float] | tuple[None, None]:
    """
        Provides the combination of entries in the PhaseDiagram that gives the
        lowest formation enthalpy with the same composition as the given entry
        excluding entries with the same composition and the energy difference
        per atom between the given entry and the energy of the combination found.

        For unstable entries that are not polymorphs of stable entries (or completely
        novel entries) this is simply the energy above (or below) the convex hull.

        For entries with the same composition as one of the stable entries in the
        phase diagram setting `stable_only` to `False` (Default) allows for entries
        not previously on the convex hull to be considered in the combination.
        In this case the energy returned is what is referred to as the decomposition
        enthalpy in:

        1. Bartel, C., Trewartha, A., Wang, Q., Dunn, A., Jain, A., Ceder, G.,
            A critical examination of compound stability predictions from
            machine-learned formation energies, npj Computational Materials 6, 97 (2020)

        For stable entries setting `stable_only` to `True` returns the same energy
        as `get_equilibrium_reaction_energy`. This function is based on a constrained
        optimization rather than recalculation of the convex hull making it
        algorithmically cheaper. However, if `tol` is too loose there is potential
        for this algorithm to converge to a different solution.

        Args:
            entry (PDEntry): A PDEntry like object.
            space_limit (int): The maximum number of competing entries to consider
                before calculating a second convex hull to reducing the complexity
                of the optimization.
            stable_only (bool): Only use stable materials as competing entries.
            tols (list[float]): Tolerances for convergence of the SLSQP optimization
                when finding the equilibrium reaction. Tighter tolerances tested first.
            maxiter (int): The maximum number of iterations of the SLSQP optimizer
                when finding the equilibrium reaction.
            **kwargs: Passed to get_decomp_and_e_above_hull.

        Returns:
            tuple[decomp, energy]: The decomposition  is given as a dict of {PDEntry, amount}
                for all entries in the decomp reaction where amount is the amount of the
                fractional composition. The phase separation energy is given per atom.
        """
    entry_frac = entry.composition.fractional_composition
    entry_elems = frozenset(entry_frac.elements)
    if entry.is_element:
        return self.get_decomp_and_e_above_hull(entry, allow_negative=True, **kwargs)
    if stable_only:
        compare_entries = self._get_stable_entries_in_space(entry_elems)
    else:
        compare_entries = [e for e, s in zip(self.qhull_entries, self._qhull_spaces) if entry_elems.issuperset(s)]
    same_comp_mem_ids = [id(c) for c in compare_entries if len(entry_frac) == len(c.composition) and all((abs(v - c.composition.get_atomic_fraction(el)) <= Composition.amount_tolerance for el, v in entry_frac.items()))]
    if not any((id(e) in same_comp_mem_ids for e in self._get_stable_entries_in_space(entry_elems))):
        return self.get_decomp_and_e_above_hull(entry, allow_negative=True, **kwargs)
    competing_entries = {c for c in compare_entries if id(c) not in same_comp_mem_ids}
    if len(competing_entries) > space_limit and (not stable_only):
        warnings.warn(f'There are {len(competing_entries)} competing entries for {entry.composition} - Calculating inner hull to discard additional unstable entries')
        reduced_space = competing_entries - {*self._get_stable_entries_in_space(entry_elems)} | {*self.el_refs.values()}
        inner_hull = PhaseDiagram(reduced_space)
        competing_entries = inner_hull.stable_entries | {*self._get_stable_entries_in_space(entry_elems)}
        competing_entries = {c for c in compare_entries if id(c) not in same_comp_mem_ids}
    if len(competing_entries) > space_limit:
        warnings.warn(f'There are {len(competing_entries)} competing entries for {entry.composition} - Using SLSQP to find decomposition likely to be slow')
    decomp = _get_slsqp_decomp(entry.composition, competing_entries, tols, maxiter)
    decomp_enthalpy = np.sum([c.energy_per_atom * amt for c, amt in decomp.items()])
    decomp_enthalpy = entry.energy_per_atom - decomp_enthalpy
    return (decomp, decomp_enthalpy)