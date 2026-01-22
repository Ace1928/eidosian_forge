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
def get_max_instability(self, min_voltage=None, max_voltage=None):
    """The maximum instability along a path for a specific voltage range.

        Args:
            min_voltage: The minimum allowable voltage.
            max_voltage: The maximum allowable voltage.

        Returns:
            Maximum decomposition energy of all compounds along the insertion
            path (a subset of the path can be chosen by the optional arguments)
        """
    data = []
    for pair in self._select_in_voltage_range(min_voltage, max_voltage):
        if getattr(pair, 'decomp_e_charge', None) is not None:
            data.append(pair.decomp_e_charge)
        if getattr(pair, 'decomp_e_discharge', None) is not None:
            data.append(pair.decomp_e_discharge)
    return max(data) if len(data) > 0 else None