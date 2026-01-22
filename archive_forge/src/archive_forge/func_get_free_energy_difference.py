import numpy as np
from ase.md.langevin import Langevin
from ase.calculators.mixing import MixedCalculator
def get_free_energy_difference(self):
    """ Return the free energy difference between calc2 and calc1, by
        integrating dH/dlam along the switching path

        Returns
        -------
        float
            Free energy difference, F2 - F1
        """
    if len(self.path_data) == 0:
        raise ValueError('No free energy data found.')
    lambdas = self.path_data[:, 1]
    U1 = self.path_data[:, 2]
    U2 = self.path_data[:, 3]
    delta_F = np.trapz(U2 - U1, lambdas)
    return delta_F