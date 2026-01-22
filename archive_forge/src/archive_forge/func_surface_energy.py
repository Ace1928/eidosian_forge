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
def surface_energy(self, ucell_entry, ref_entries=None):
    """
        Calculates the surface energy of this SlabEntry.

        Args:
            ucell_entry (entry): An entry object for the bulk
            ref_entries (list: [entry]): A list of entries for each type
                of element to be used as a reservoir for non-stoichiometric
                systems. The length of this list MUST be n-1 where n is the
                number of different elements in the bulk entry. The chempot
                of the element ref_entry that is not in the list will be
                treated as a variable.

        Returns:
            float: The surface energy of the slab.
        """
    ref_entries = ref_entries or []
    slab_comp = self.composition.as_dict()
    ucell_entry_comp = ucell_entry.composition.reduced_composition.as_dict()
    slab_clean_comp = Composition({el: slab_comp[el] for el in ucell_entry_comp})
    if slab_clean_comp.reduced_composition != ucell_entry.composition.reduced_composition:
        list_els = [next(iter(entry.composition.as_dict())) for entry in ref_entries]
        if not any((el in list_els for el in ucell_entry.composition.as_dict())):
            warnings.warn('Elemental references missing for the non-dopant species.')
    gamma = (Symbol('E_surf') - Symbol('Ebulk')) / (2 * Symbol('A'))
    ucell_comp = ucell_entry.composition
    ucell_reduced_comp = ucell_comp.reduced_composition
    ref_entries_dict = {str(next(iter(ref.composition.as_dict()))): ref for ref in ref_entries}
    ref_entries_dict.update(self.ads_entries_dict)
    gibbs_bulk = ucell_entry.energy / ucell_comp.get_integer_formula_and_factor()[1]
    bulk_energy, gbulk_eqn = (0, 0)
    for el, ref in ref_entries_dict.items():
        N, delu = (self.composition.as_dict()[el], Symbol(f'delu_{el}'))
        if el in ucell_comp.as_dict():
            gbulk_eqn += ucell_reduced_comp[el] * (delu + ref.energy_per_atom)
        bulk_energy += N * (Symbol('delu_' + el) + ref.energy_per_atom)
    for ref_el in ucell_comp.as_dict():
        if str(ref_el) not in ref_entries_dict:
            break
    ref_e_per_a = (gibbs_bulk - gbulk_eqn) / ucell_reduced_comp.as_dict()[ref_el]
    bulk_energy += self.composition.as_dict()[ref_el] * ref_e_per_a
    se = gamma.subs({Symbol('E_surf'): self.energy, Symbol('Ebulk'): bulk_energy, Symbol('A'): self.surface_area})
    return float(se) if type(se).__name__ == 'Float' else se