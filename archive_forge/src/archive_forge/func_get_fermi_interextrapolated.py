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
def get_fermi_interextrapolated(self, concentration: float, temperature: float, warn: bool=True, c_ref: float=10000000000.0, **kwargs) -> float:
    """Similar to get_fermi except that when get_fermi fails to converge,
        an interpolated or extrapolated fermi is returned with the assumption
        that the Fermi level changes linearly with log(abs(concentration)).

        Args:
            concentration: The doping concentration in 1/cm^3. Negative values
                represent n-type doping and positive values represent p-type
                doping.
            temperature: The temperature in Kelvin.
            warn: Whether to give a warning the first time the fermi cannot be
                found.
            c_ref: A doping concentration where get_fermi returns a
                value without error for both c_ref and -c_ref.
            **kwargs: Keyword arguments passed to the get_fermi function.

        Returns:
            The Fermi level. Note, the value is possibly interpolated or
            extrapolated and must be used with caution.
        """
    try:
        return self.get_fermi(concentration, temperature, **kwargs)
    except ValueError as exc:
        if warn:
            warnings.warn(str(exc))
        if abs(concentration) < c_ref:
            if abs(concentration) < 1e-10:
                concentration = 1e-10
            f2 = self.get_fermi_interextrapolated(max(10, abs(concentration) * 10.0), temperature, warn=False, **kwargs)
            f1 = self.get_fermi_interextrapolated(-max(10, abs(concentration) * 10.0), temperature, warn=False, **kwargs)
            c2 = np.log(abs(1 + self.get_doping(f2, temperature)))
            c1 = -np.log(abs(1 + self.get_doping(f1, temperature)))
            slope = (f2 - f1) / (c2 - c1)
            return f2 + slope * (np.sign(concentration) * np.log(abs(1 + concentration)) - c2)
        f_ref = self.get_fermi_interextrapolated(np.sign(concentration) * c_ref, temperature, warn=False, **kwargs)
        f_new = self.get_fermi_interextrapolated(concentration / 10.0, temperature, warn=False, **kwargs)
        clog = np.sign(concentration) * np.log(abs(concentration))
        c_new_log = np.sign(concentration) * np.log(abs(self.get_doping(f_new, temperature)))
        slope = (f_new - f_ref) / (c_new_log - np.sign(concentration) * 10.0)
        return f_new + slope * (clog - c_new_log)