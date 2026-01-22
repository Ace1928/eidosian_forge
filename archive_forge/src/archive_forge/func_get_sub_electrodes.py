from __future__ import annotations
import itertools
from dataclasses import dataclass
from typing import TYPE_CHECKING
from monty.json import MontyDecoder
from scipy.constants import N_A
from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from pymatgen.apps.battery.battery_abc import AbstractElectrode, AbstractVoltagePair
from pymatgen.core import Composition, Element
from pymatgen.core.units import Charge, Time
from pymatgen.entries.computed_entries import ComputedEntry, ComputedStructureEntry
def get_sub_electrodes(self, adjacent_only=True, include_myself=True):
    """If this electrode contains multiple voltage steps, then it is possible
        to use only a subset of the voltage steps to define other electrodes.
        For example, an LiTiO2 electrode might contain three subelectrodes:
        [LiTiO2 --> TiO2, LiTiO2 --> Li0.5TiO2, Li0.5TiO2 --> TiO2]
        This method can be used to return all the subelectrodes with some
        options.

        Args:
            adjacent_only: Only return electrodes from compounds that are
                adjacent on the convex hull, i.e. no electrodes returned
                will have multiple voltage steps if this is set True.
            include_myself: Include this identical electrode in the list of
                results.

        Returns:
            A list of InsertionElectrode objects
        """
    battery_list = []
    pair_it = self.voltage_pairs if adjacent_only else itertools.combinations_with_replacement(self.voltage_pairs, 2)
    ion = self.working_ion
    for pair in pair_it:
        entry_charge = pair.entry_charge if adjacent_only else pair[0].entry_charge
        entry_discharge = pair.entry_discharge if adjacent_only else pair[1].entry_discharge

        def in_range(entry):
            chg_frac = entry_charge.composition.get_atomic_fraction(ion)
            dischg_frac = entry_discharge.composition.get_atomic_fraction(ion)
            frac = entry.composition.get_atomic_fraction(ion)
            return chg_frac <= frac <= dischg_frac
        if include_myself or entry_charge != self.fully_charged_entry or entry_discharge != self.fully_discharged_entry:
            unstable_entries = filter(in_range, self.get_unstable_entries())
            stable_entries = filter(in_range, self.get_stable_entries())
            all_entries = list(stable_entries)
            all_entries.extend(unstable_entries)
            battery_list.append(type(self).from_entries(all_entries, self.working_ion_entry))
    return battery_list