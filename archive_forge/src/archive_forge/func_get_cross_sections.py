import sys
import numpy as np
import ase.units as u
from ase.parallel import world, paropen, parprint
from ase.vibrations import Vibrations
from ase.vibrations.raman import Raman, RamanCalculatorBase
def get_cross_sections(self, omega, gamma):
    """Returns Raman cross sections for each vibration."""
    I_v = self.intensity(omega, gamma)
    pre = 1.0 / 16 / np.pi ** 2 / u._eps0 ** 2 / u._c ** 4
    omS_v = omega - self.om_Q
    return pre * omega * omS_v ** 3 * I_v