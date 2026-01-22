import numpy as np
import ase.units as un
def calculate(self, atoms):
    """
        Calculate the polarizability for frequency omega

        Parameters
        ----------
        atoms: atoms class
            The atoms definition of the system. Not used but required by Raman
            calculator
        """
    pmat = self.get_polarizability(self.omega, Eext=np.array([1.0, 1.0, 1.0]))
    return pmat[:, :, 0].real * un.Bohr ** 2 / un.Ha