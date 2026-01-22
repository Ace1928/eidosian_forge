from __future__ import annotations
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING
from monty.json import MSONable
from scipy.constants import N_A
from pymatgen.core import Composition, Element
def get_specific_energy(self, min_voltage=None, max_voltage=None, use_overall_normalization=True):
    """Returns the specific energy of the battery in mAh/g.

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
            Specific energy in Wh/kg across the insertion path (a subset of
            the path can be chosen by the optional arguments)
        """
    return self.get_capacity_grav(min_voltage, max_voltage, use_overall_normalization) * self.get_average_voltage(min_voltage, max_voltage)