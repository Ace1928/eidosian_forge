from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
import scipy.constants as const
from monty.functools import lazy_property
from monty.json import MSONable
from scipy.ndimage import gaussian_filter1d
from pymatgen.core.structure import Structure
from pymatgen.util.coord import get_linear_interpolated_value
def mae(self, other: PhononDos, two_sided: bool=True) -> float:
    """Mean absolute error between two DOSs.

        Args:
            other (PhononDos): Another phonon DOS
            two_sided (bool): Whether to calculate the two-sided MAE meaning interpolate each DOS to the
                other's frequencies and averaging the two MAEs. Defaults to True.

        Returns:
            float: Mean absolute error.
        """
    self_interpolated = np.interp(self.frequencies, other.frequencies, other.densities)
    self_mae = np.abs(self.densities - self_interpolated).mean()
    if two_sided:
        other_interpolated = np.interp(other.frequencies, self.frequencies, self.densities)
        other_mae = np.abs(other.densities - other_interpolated).mean()
        return (self_mae + other_mae) / 2
    return self_mae