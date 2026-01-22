from __future__ import annotations
import functools
import warnings
from collections import namedtuple
from typing import TYPE_CHECKING, NamedTuple
import numpy as np
from monty.json import MSONable
from scipy.constants import value as _cd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import hilbert
from pymatgen.core import Structure, get_el_sp
from pymatgen.core.spectrum import Spectrum
from pymatgen.electronic_structure.core import Orbital, OrbitalType, Spin
from pymatgen.util.coord import get_linear_interpolated_value
def get_fermi(self, concentration: float, temperature: float, rtol: float=0.01, nstep: int=50, step: float=0.1, precision: int=8) -> float:
    """Finds the Fermi level at which the doping concentration at the given
        temperature (T) is equal to concentration. A greedy algorithm is used
        where the relative error is minimized by calculating the doping at a
        grid which continually becomes finer.

        Args:
            concentration: The doping concentration in 1/cm^3. Negative values
                represent n-type doping and positive values represent p-type
                doping.
            temperature: The temperature in Kelvin.
            rtol: The maximum acceptable relative error.
            nstep: The number of steps checked around a given Fermi level.
            step: Initial step in energy when searching for the Fermi level.
            precision: Essentially the decimal places of calculated Fermi level.

        Raises:
            ValueError: If the Fermi level cannot be found.

        Returns:
            The Fermi level in eV. Note that this is different from the default
            dos.efermi.
        """
    fermi = self.efermi
    relative_error = [float('inf')]
    for _ in range(precision):
        fermi_range = np.arange(-nstep, nstep + 1) * step + fermi
        calc_doping = np.array([self.get_doping(fermi_lvl, temperature) for fermi_lvl in fermi_range])
        relative_error = np.abs(calc_doping / concentration - 1.0)
        fermi = fermi_range[np.argmin(relative_error)]
        step /= 10.0
    if min(relative_error) > rtol:
        raise ValueError(f'Could not find fermi within {rtol:.1%} of concentration={concentration!r}')
    return fermi