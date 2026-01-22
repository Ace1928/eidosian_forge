from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
import scipy.constants as const
from monty.functools import lazy_property
from monty.json import MSONable
from scipy.ndimage import gaussian_filter1d
from pymatgen.core.structure import Structure
from pymatgen.util.coord import get_linear_interpolated_value
def get_last_peak(self, threshold: float=0.05) -> float:
    """Find the last peak in the phonon DOS defined as the highest frequency with a DOS
        value at least threshold * height of the overall highest DOS peak.
        A peak is any local maximum of the DOS as a function of frequency.
        Use dos.get_interpolated_value(peak_freq) to get density at peak_freq.

        TODO method added by @janosh on 2023-12-18. seems to work in most cases but
        was not extensively tested. PRs with improvements welcome!

        Args:
            threshold (float, optional): Minimum ratio of the height of the last peak
                to the height of the highest peak. Defaults to 0.05 = 5%. In case no peaks
                are high enough to match, the threshold is reset to half the height of the
                second-highest peak.

        Returns:
            float: last DOS peak frequency (in THz)
        """
    first_deriv = np.gradient(self.densities, self.frequencies)
    second_deriv = np.gradient(first_deriv, self.frequencies)
    maxima = (first_deriv[:-1] > 0) & (first_deriv[1:] < 0) & (second_deriv[:-1] < 0)
    maxima_freqs = (self.frequencies[:-1][maxima] + self.frequencies[1:][maxima]) / 2
    max_dos = max(self.densities)
    threshold = threshold * max_dos
    filtered_maxima_freqs = maxima_freqs[self.densities[:-1][maxima] >= threshold]
    if len(filtered_maxima_freqs) == 0:
        second_highest_peak = sorted(self.densities)[-2]
        threshold = second_highest_peak / 2
        filtered_maxima_freqs = maxima_freqs[self.densities[:-1][maxima] >= threshold]
    return max(filtered_maxima_freqs)