from __future__ import annotations
from math import isclose
from typing import TYPE_CHECKING
import numpy as np
import scipy.constants as const
from pymatgen.core.units import kb as kb_ev
from pymatgen.util.due import Doi, due
@classmethod
def from_gaussian_output(cls, output: GaussianOutput, **kwargs) -> Self:
    """
        Args:
            output (GaussianOutput): Pymatgen GaussianOutput object

        Returns:
            QuasiRRHO: QuasiRRHO class instantiated from a Gaussian Output
        """
    mult = output.spin_multiplicity
    elec_e = output.final_energy
    mol = output.final_structure
    vib_freqs = [freq['frequency'] for freq in output.frequencies[-1]]
    return cls(mol=mol, frequencies=vib_freqs, energy=elec_e, mult=mult, **kwargs)