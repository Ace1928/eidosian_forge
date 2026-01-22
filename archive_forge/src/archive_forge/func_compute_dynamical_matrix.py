from math import pi, sqrt
import warnings
from pathlib import Path
import numpy as np
import numpy.linalg as la
import numpy.fft as fft
import ase
import ase.units as units
from ase.parallel import world
from ase.dft import monkhorst_pack
from ase.io.trajectory import Trajectory
from ase.utils.filecache import MultiFileJSONCache
def compute_dynamical_matrix(self, q_scaled: np.ndarray, D_N: np.ndarray):
    """ Computation of the dynamical matrix in momentum space D_ab(q).
            This is a Fourier transform from real-space dynamical matrix D_N
            for a given momentum vector q.

        q_scaled: q vector in scaled coordinates.

        D_N: the dynamical matrix in real-space. It is necessary, at least
             currently, to provide this matrix explicitly (rather than use
             self.D_N) because this matrix is modified by the Born charges
             contributions and these modifications are momentum (q) dependent.

        Result:
            D(q): two-dimensional, complex-valued array of
                  shape=(3 * natoms, 3 * natoms).
        """
    R_cN = self._lattice_vectors_array
    phase_N = np.exp(-2j * pi * np.dot(q_scaled, R_cN))
    D_q = np.sum(phase_N[:, np.newaxis, np.newaxis] * D_N, axis=0)
    return D_q