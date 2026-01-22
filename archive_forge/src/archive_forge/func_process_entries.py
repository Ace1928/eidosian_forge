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
def process_entries(self, entries: AnyComputedEntry | list[AnyComputedEntry], clean: bool=True, verbose: bool=True, inplace: bool=True, mixing_state_data=None) -> list[AnyComputedEntry]:
    """Process a sequence of entries with the DFT mixing scheme. Note
        that this method will change the data of the original entries.

        Args:
            entries: ComputedEntry or [ComputedEntry]. Pass all entries as a single list, even if they are
                computed with different functionals or require different preprocessing. This list will
                automatically be filtered based on run_type_1 and run_type_2, and processed according to
                compat_1 and compat_2.

                Note that under typical use, when mixing_state_data=None, the entries MUST be
                ComputedStructureEntry. They will be matched using structure_matcher.
            clean (bool): Whether to remove any previously-applied energy adjustments.
                If True, all EnergyAdjustment are removed prior to processing the Entry.
                Default is True.
            verbose (bool): Whether to print verbose error messages about the mixing scheme. Default is True.
            inplace (bool): Whether to adjust input entries in place. Default is True.
            mixing_state_data: A DataFrame containing information about which Entries
                correspond to the same materials, which are stable on the phase diagrams of
                the respective run_types, etc. If None (default), it will be generated from the
                list of entries using MaterialsProjectDFTMixingScheme.get_mixing_state_data.
                This argument is included to facilitate use of the mixing scheme in high-throughput
                databases where an alternative to get_mixing_state_data is desirable for performance
                reasons. In general, it should always be left at the default value (None) to avoid
                inconsistencies between the mixing state data and the properties of the
                ComputedStructureEntry in entries.

        Returns:
            list[AnyComputedEntry]: Adjusted entries. Entries in the original list incompatible with
                chosen correction scheme are excluded from the returned list.
        """
    processed_entry_list: list = []
    if len(entries) == 1:
        warnings.warn(f'{type(self).__name__} cannot process single entries. Supply a list of entries.')
        return processed_entry_list
    if not inplace:
        entries = copy.deepcopy(entries)
    if clean:
        for entry in entries:
            for ea in entry.energy_adjustments:
                entry.energy_adjustments.remove(ea)
    entries_type_1, entries_type_2 = self._filter_and_sort_entries(entries, verbose=verbose)
    if mixing_state_data is None:
        if verbose:
            print('  Generating mixing state data from provided entries.')
        mixing_state_data = self.get_mixing_state_data(entries_type_1 + entries_type_2)
    if verbose:
        hull_entries_2 = 0
        stable_df = mixing_state_data[mixing_state_data['is_stable_1']]
        if len(stable_df) > 0:
            hull_entries_2 = sum(stable_df['energy_2'].notna())
        print(f'  Entries contain {self.run_type_2} calculations for {hull_entries_2} of {len(stable_df)} {self.run_type_1} hull entries.')
        if hull_entries_2 == len(stable_df):
            print(f'  {self.run_type_1} energies will be adjusted to the {self.run_type_2} scale')
        else:
            print(f'  {self.run_type_2} energies will be adjusted to the {self.run_type_1} scale')
        if hull_entries_2 > 0:
            print(f'  The energy above hull for {self.run_type_2} materials at compositions with {self.run_type_2} hull entries will be preserved. For other compositions, Energies of {self.run_type_2} materials will be set equal to those of matching {self.run_type_1} materials')
    for entry in entries_type_1 + entries_type_2:
        ignore_entry = False
        try:
            adjustments = self.get_adjustments(entry, mixing_state_data)
        except CompatibilityError as exc:
            if 'WARNING!' in str(exc):
                warnings.warn(str(exc))
            elif verbose:
                print(f'  {exc}')
            ignore_entry = True
            continue
        for ea in adjustments:
            if (ea.name, ea.cls, ea.value) in [(ea2.name, ea2.cls, ea2.value) for ea2 in entry.energy_adjustments]:
                pass
            elif (ea.name, ea.cls) in [(ea2.name, ea2.cls) for ea2 in entry.energy_adjustments]:
                ignore_entry = True
                warnings.warn(f'Entry {entry.entry_id} already has an energy adjustment called {ea.name}, but its value differs from the value of {ea.value:.3f} calculated here. This Entry will be discarded.')
            else:
                entry.energy_adjustments.append(ea)
        if not ignore_entry:
            processed_entry_list.append(entry)
    if verbose:
        count_type_1 = len([e for e in processed_entry_list if e.parameters['run_type'] in self.valid_rtypes_1])
        count_type_2 = len([e for e in processed_entry_list if e.parameters['run_type'] in self.valid_rtypes_2])
        print(f'\nProcessing complete. Mixed entries contain {count_type_1} {self.run_type_1} and {count_type_2} {self.run_type_2} entries.\n')
        self.display_entries(processed_entry_list)
    return processed_entry_list