from __future__ import annotations
import itertools
from dataclasses import dataclass
from typing import TYPE_CHECKING
import numpy as np
import scipy.constants
import scipy.special
from monty.json import MSONable
from tqdm import tqdm
from pymatgen.electronic_structure.core import Spin
from pymatgen.io.vasp.outputs import Vasprun, Waveder
def plot_weighted_transition_data(self, idir: int, jdir: int, mask: NDArray | None=None, min_val: float=0.0):
    """Data for plotting the weight matrix elements as a scatter plot.

        Since the computation of the final spectrum (especially the smearing part)
        is still fairly expensive.  This function can be used to check the values
        of some portion of the spectrum (defined by the mask).
        In a sense, we are lookin at the imaginary part of the dielectric function
        before the smearing is applied.

        Args:
            idir: First direction of the dielectric tensor.
            jdir: Second direction of the dielectric tensor.
            mask: Mask to apply to the CDER for the bands/kpoint/spin
                index to include in the calculation
            min_val: Minimum value below this value the matrix element will not be shown.
        """
    cderm = self.cder * mask if mask is not None else self.cder
    norm_kweights = np.array(self.kweights) / np.sum(self.kweights)
    eigs_shifted = self.eigs - self.efermi
    rspin = 3 - cderm.shape[3]
    try:
        min_band0, max_band0 = (np.min(np.where(cderm)[0]), np.max(np.where(cderm)[0]))
        min_band1, max_band1 = (np.min(np.where(cderm)[1]), np.max(np.where(cderm)[1]))
    except ValueError as exc:
        if 'zero-size array' in str(exc):
            raise ValueError('No matrix elements found. Check the mask.')
        raise
    x_val = []
    y_val = []
    text = []
    _, _, nk, nspin = cderm.shape[:4]
    iter_idx = [range(min_band0, max_band0 + 1), range(min_band1, max_band1 + 1), range(nk), range(nspin)]
    num_ = (max_band0 - min_band0) * (max_band1 - min_band1) * nk * nspin
    for ib, jb, ik, ispin in tqdm(itertools.product(*iter_idx), total=num_):
        fermi_w_i = step_func(eigs_shifted[ib, ik, ispin] / self.sigma, self.ismear)
        fermi_w_j = step_func(eigs_shifted[jb, ik, ispin] / self.sigma, self.ismear)
        weight = (fermi_w_j - fermi_w_i) * rspin * norm_kweights[ik]
        A = cderm[ib, jb, ik, ispin, idir] * np.conjugate(cderm[ib, jb, ik, ispin, jdir])
        decel = self.eigs[jb, ik, ispin] - self.eigs[ib, ik, ispin]
        matrix_el = np.abs(A) * float(weight)
        if matrix_el > min_val:
            x_val.append(decel)
            y_val.append(matrix_el)
            text.append(f's:{ispin}, k:{ik}, {ib} -> {jb} ({decel:.2f})')
    return (x_val, y_val, text)