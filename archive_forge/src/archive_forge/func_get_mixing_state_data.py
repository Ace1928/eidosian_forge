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
def get_mixing_state_data(self, entries: list[ComputedStructureEntry]):
    """Generate internal state data to be passed to get_adjustments.

        Args:
            entries: The list of ComputedStructureEntry to process. It is assumed that the entries have
                already been filtered using _filter_and_sort_entries() to remove any irrelevant run types,
                apply compat_1 and compat_2, and confirm that all have unique entry_id.

        Returns:
            DataFrame: A pandas DataFrame that contains information associating structures from
                different functionals with specific materials and establishing how many run_type_1
                ground states have been computed with run_type_2. The DataFrame contains one row
                for each distinct material (Structure), with the following columns:
                    formula: str the reduced_formula
                    spacegroup: int the spacegroup
                    num_sites: int the number of sites in the Structure
                    entry_id_1: the entry_id of the run_type_1 entry
                    entry_id_2: the entry_id of the run_type_2 entry
                    run_type_1: Optional[str] the run_type_1 value
                    run_type_2: Optional[str] the run_type_2 value
                    energy_1: float or nan the ground state energy in run_type_1 in eV/atom
                    energy_2: float or nan the ground state energy in run_type_2 in eV/atom
                    is_stable_1: bool whether this material is stable on the run_type_1 PhaseDiagram
                    hull_energy_1: float or nan the energy of the run_type_1 hull at this composition in eV/atom
                    hull_energy_2: float or nan the energy of the run_type_1 hull at this composition in eV/atom
            None: Returns None if the supplied ComputedStructureEntry are insufficient for applying
                the mixing scheme.
        """
    filtered_entries = []
    for entry in entries:
        if not isinstance(entry, ComputedStructureEntry):
            warnings.warn(f'Entry {entry.entry_id} is not a ComputedStructureEntry and will be ignored. The DFT mixing scheme requires structures for all entries')
            continue
        filtered_entries.append(entry)
    entries_type_1 = [e for e in filtered_entries if e.parameters['run_type'] in self.valid_rtypes_1]
    entries_type_2 = [e for e in filtered_entries if e.parameters['run_type'] in self.valid_rtypes_2]
    pd_type_1, pd_type_2 = (None, None)
    try:
        pd_type_1 = PhaseDiagram(entries_type_1)
    except ValueError:
        warnings.warn(f'{self.run_type_1} entries do not form a complete PhaseDiagram.')
    try:
        pd_type_2 = PhaseDiagram(entries_type_2)
    except ValueError:
        warnings.warn(f'{self.run_type_2} entries do not form a complete PhaseDiagram.')
    all_entries = list(entries_type_1) + list(entries_type_2)
    row_list = []
    columns = ['formula', 'spacegroup', 'num_sites', 'is_stable_1', 'entry_id_1', 'entry_id_2', 'run_type_1', 'run_type_2', 'energy_1', 'energy_2', 'hull_energy_1', 'hull_energy_2']

    def _get_sg(struct) -> int:
        """Helper function to get spacegroup with a loose tolerance."""
        try:
            return struct.get_space_group_info(symprec=0.1)[1]
        except Exception:
            return -1
    structures = []
    for entry in all_entries:
        struct = entry.structure
        struct.entry_id = entry.entry_id
        structures.append(struct)
    for comp, comp_group in groupby(sorted(structures, key=lambda s: s.composition), key=lambda s: s.composition):
        l_comp_group = list(comp_group)
        for sg, pre_group in groupby(sorted(l_comp_group, key=_get_sg), key=_get_sg):
            l_pre_group = list(pre_group)
            if comp.reduced_formula in ['O2', 'H2', 'Cl2', 'F2', 'N2', 'I', 'Br', 'H2O'] and self.fuzzy_matching:
                for idx, site_group in groupby(sorted(l_pre_group, key=len), key=len):
                    l_sitegroup = list(site_group)
                    row_list.append(self._populate_df_row(l_sitegroup, comp, sg, idx, pd_type_1, pd_type_2, all_entries))
            else:
                for group in self.structure_matcher.group_structures(l_pre_group):
                    group = list(group)
                    idx = len(group[0])
                    row_list.append(self._populate_df_row(group, comp, sg, idx, pd_type_1, pd_type_2, all_entries))
    mixing_state_data = pd.DataFrame(row_list, columns=columns)
    return mixing_state_data.sort_values(['formula', 'energy_1', 'spacegroup', 'num_sites'], ignore_index=True)