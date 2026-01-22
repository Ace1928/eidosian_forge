import sys
import weakref
import numpy as np
from ase.md.md import MolecularDynamics
from ase import units
def set_temperature(self, temperature=None, *, temperature_K=None):
    """Set the temperature.

        Parameters:

        temperature: float (deprecated)
            The new temperature in eV.  Deprecated, use ``temperature_K``.

        temperature_K: float (keyword-only argument)
            The new temperature, in K.
        """
    self.temperature = units.kB * self._process_temperature(temperature, temperature_K, 'eV')
    self._calculateconstants()