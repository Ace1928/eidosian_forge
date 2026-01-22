from __future__ import annotations
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING
from monty.json import MSONable
from scipy.constants import N_A
from pymatgen.core import Composition, Element
def get_capacity_grav(self, min_voltage=None, max_voltage=None, use_overall_normalization=True):
    """Get the gravimetric capacity of the electrode.

        Args:
            min_voltage (float): The minimum allowable voltage for a given
                step.
            max_voltage (float): The maximum allowable voltage allowable for a
                given step.
            use_overall_normalization (booL): If False, normalize by the
                discharged state of only the voltage pairs matching the voltage
                criteria. if True, use default normalization of the full
                electrode path.

        Returns:
            Gravimetric capacity in mAh/g across the insertion path (a subset
            of the path can be chosen by the optional arguments).
        """
    pairs_in_range = self._select_in_voltage_range(min_voltage, max_voltage)
    normalization_mass = self.normalization_mass if use_overall_normalization or len(pairs_in_range) == 0 else pairs_in_range[-1].mass_discharge
    return sum((pair.mAh for pair in pairs_in_range)) / normalization_mass