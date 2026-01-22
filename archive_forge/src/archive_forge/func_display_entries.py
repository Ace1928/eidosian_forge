from __future__ import annotations
import copy
import os
import warnings
from itertools import groupby
import numpy as np
import pandas as pd
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.entries.compatibility import (
from pymatgen.entries.computed_entries import ComputedStructureEntry, ConstantEnergyAdjustment
from pymatgen.entries.entry_tools import EntrySet
@staticmethod
def display_entries(entries):
    """Generate a pretty printout of key properties of a list of ComputedEntry."""
    entries = sorted(entries, key=lambda e: (e.reduced_formula, e.energy_per_atom))
    try:
        pd = PhaseDiagram(entries)
    except ValueError:
        return
    print(f'{'entry_id':<12}{'formula':<12}{'spacegroup':<12}{'run_type':<10}{'eV/atom':<8}{'corr/atom':<9} {'e_above_hull':<9}')
    for entry in entries:
        print(f'{entry.entry_id:<12}{entry.reduced_formula:<12}{entry.structure.get_space_group_info()[0]:<12}{entry.parameters['run_type']:<10}{entry.energy_per_atom:<8.3f}{entry.correction / entry.composition.num_atoms:<9.3f} {pd.get_e_above_hull(entry):<9.3f}')
    return