import numpy as np
import ase.units as un
def get_polarizability(self, omega, Eext=np.array([1.0, 1.0, 1.0]), inter=True):
    """
        Calculate the polarizability of a molecule via linear response TDDFT
        calculation.

        Parameters
        ----------
        omega: float or array like
            frequency range for which the polarizability should be computed, in eV

        Returns
        -------
        polarizability: array like (complex)
            array of dimension (3, 3, nff) with nff the number of frequency,
            the first and second dimension are the matrix elements of the
            polarizability in atomic units::

                P_xx, P_xy, P_xz, Pyx, .......

        Example
        -------

        from ase.calculators.siesta.siesta_lrtddft import siestaLRTDDFT
        from ase.build import molecule
        import numpy as np
        import matplotlib.pyplot as plt

        # Define the systems
        CH4 = molecule('CH4')

        lr = siestaLRTDDFT(label="siesta", jcutoff=7, iter_broadening=0.15,
                            xc_code='LDA,PZ', tol_loc=1e-6, tol_biloc=1e-7)

        # run DFT calculation with Siesta
        lr.get_ground_state(CH4)

        # run TDDFT calculation with PyNAO
        freq=np.arange(0.0, 25.0, 0.05)
        pmat = lr.get_polarizability(freq) 
        """
    from pynao import tddft_iter
    if not self.initialize:
        self.tddft = tddft_iter(**self.lrtddft_params)
    if isinstance(omega, float):
        freq = np.array([omega])
    elif isinstance(omega, list):
        freq = np.array([omega])
    elif isinstance(omega, np.ndarray):
        freq = omega
    else:
        raise ValueError('omega soulf')
    freq_cmplx = freq / un.Ha + 1j * self.tddft.eps
    if inter:
        pmat = -self.tddft.comp_polariz_inter_Edir(freq_cmplx, Eext=Eext)
        self.dn = self.tddft.dn
    else:
        pmat = -self.tddft.comp_polariz_nonin_Edir(freq_cmplx, Eext=Eext)
        self.dn = self.tddft.dn0
    return pmat