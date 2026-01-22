from __future__ import annotations
import abc
import json
import math
import os
import warnings
from itertools import combinations
from typing import TYPE_CHECKING, Literal, cast
import numpy as np
from monty.json import MontyDecoder, MontyEncoder, MSONable
from scipy.interpolate import interp1d
from uncertainties import ufloat
from pymatgen.core.composition import Composition
from pymatgen.entries import Entry
from pymatgen.util.due import Doi, due
@due.dcite(Doi('10.1038/s41467-018-06682-4', 'Gibbs free energy SISSO descriptor'))
def gf_sisso(self) -> float:
    """Gibbs Free Energy of formation as calculated by SISSO descriptor from Bartel
        et al. (2018). Units: eV (not normalized).

        WARNING: This descriptor only applies to solids. The implementation here
        attempts to detect and use downloaded NIST-JANAF data for common
        experimental gases (e.g. CO2) where possible. Note that experimental data is
        only for Gibbs Free Energy of formation, so expt. entries will register as
        having a formation enthalpy of 0.

        Reference: Bartel, C. J., Millican, S. L., Deml, A. M., Rumptz, J. R.,
        Tumas, W., Weimer, A. W., … Holder, A. M. (2018). Physical descriptor for
        the Gibbs energy of inorganic crystalline solids and
        temperature-dependent materials chemistry. Nature Communications, 9(1),
        4168. https://doi.org/10.1038/s41467-018-06682-4

        Returns:
            float: the difference between formation enthalpy (T=0 K, Materials
            Project) and the predicted Gibbs free energy of formation  (eV)
        """
    comp = self.composition
    if comp.is_element:
        return 0
    integer_formula, factor = comp.get_integer_formula_and_factor()
    if self.experimental:
        data = G_GASES[integer_formula]
        if self.interpolated:
            g_interp = interp1d([int(t) for t in data], list(data.values()))
            energy = g_interp(self.temp)
        else:
            energy = data[str(self.temp)]
        gibbs_energy = energy * factor
    else:
        n_atoms = len(self.structure)
        vol_per_atom = self.structure.volume / n_atoms
        reduced_mass = self._reduced_mass(self.structure)
        gibbs_energy = comp.num_atoms * (self.formation_enthalpy_per_atom + self._g_delta_sisso(vol_per_atom, reduced_mass, self.temp)) - self._sum_g_i()
    return gibbs_energy