from __future__ import annotations
import math
import warnings
from typing import Literal
import numpy as np
from scipy.interpolate import interp1d
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.spectrum import Spectrum
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
def stitch(self, other: XAS, num_samples: int=500, mode: Literal['XAFS', 'L23']='XAFS') -> XAS:
    """
        Stitch XAS objects to get the full XAFS spectrum or L23 edge XANES
        spectrum depending on the mode.

        1. Use XAFS mode for stitching XANES and EXAFS with same absorption edge.
            The stitching will be performed based on wavenumber, k.
            for k <= 3, XAS(k) = XAS[XANES(k)]
            for 3 < k < max(xanes_k), will interpolate according to
                XAS(k)=f(k)*mu[XANES(k)]+(1-f(k))*mu[EXAFS(k)]
                where f(k)=cos^2((pi/2) (k-3)/(max(xanes_k)-3)
            for k > max(xanes_k), XAS(k) = XAS[EXAFS(k)]
        2. Use L23 mode for stitching L2 and L3 edge XANES for elements with
            atomic number <=30.

        Args:
            other: Another XAS object.
            num_samples (int): Number of samples for interpolation.
            mode("XAFS" | "L23"): Either XAFS mode for stitching XANES and EXAFS
                or L23 mode for stitching L2 and L3.

        Returns:
            XAS object: The stitched spectrum.
        """
    matcher = StructureMatcher()
    if not matcher.fit(self.structure, other.structure):
        raise ValueError('The input structures for spectra mismatch')
    if not self.absorbing_element == other.absorbing_element:
        raise ValueError('The absorbing elements for spectra are different')
    if not self.absorbing_index == other.absorbing_index:
        raise ValueError('The absorbing indexes for spectra are different')
    if mode == 'XAFS':
        if not self.edge == other.edge:
            raise ValueError('Only spectrum with the same absorption edge can be stitched in XAFS mode.')
        if self.spectrum_type == other.spectrum_type:
            raise ValueError('Need one XANES and one EXAFS spectrum to stitch in XAFS mode')
        xanes = self if self.spectrum_type == 'XANES' else other
        exafs = self if self.spectrum_type == 'EXAFS' else other
        if max(xanes.x) < min(exafs.x):
            raise ValueError('Energy overlap between XANES and EXAFS is needed for stitching')
        wavenumber: list[float] = []
        mu: list[float] = []
        idx = xanes.k.index(min(self.k, key=lambda x: (abs(x - 3), x)))
        mu.extend(xanes.y[:idx])
        wavenumber.extend(xanes.k[:idx])
        fs: list[float] = []
        ks = np.linspace(3, max(xanes.k), 50)
        for k in ks:
            f = np.cos(math.pi / 2 * (k - 3) / (max(xanes.k) - 3)) ** 2
            fs.append(f)
        f_xanes = interp1d(np.asarray(xanes.k), np.asarray(xanes.y), bounds_error=False, fill_value=0)
        f_exafs = interp1d(np.asarray(exafs.k), np.asarray(exafs.y), bounds_error=False, fill_value=0)
        mu_xanes = f_xanes(ks)
        mu_exafs = f_exafs(ks)
        mus = [fs[i] * mu_xanes[i] + (1 - fs[i]) * mu_exafs[i] for i in np.arange(len(ks))]
        mu.extend(mus)
        wavenumber.extend(ks)
        idx = exafs.k.index(min(exafs.k, key=lambda x: abs(x - max(xanes.k))))
        mu.extend(exafs.y[idx:])
        wavenumber.extend(exafs.k[idx:])
        f_final = interp1d(np.asarray(wavenumber), np.asarray(mu), bounds_error=False, fill_value=0)
        wavenumber_final = np.linspace(min(wavenumber), max(wavenumber), num=num_samples)
        mu_final = f_final(wavenumber_final)
        energy_final = [3.8537 * i ** 2 + xanes.e0 if i > 0 else -3.8537 * i ** 2 + xanes.e0 for i in wavenumber_final]
        return XAS(energy_final, mu_final, self.structure, self.absorbing_element, xanes.edge, 'XAFS')
    if mode == 'L23':
        if self.spectrum_type != 'XANES' or other.spectrum_type != 'XANES':
            raise ValueError('Only XANES spectrum can be stitched in L23 mode.')
        if self.edge not in ['L2', 'L3'] or other.edge not in ['L2', 'L3'] or self.edge == other.edge:
            raise ValueError('Need one L2 and one L3 edge spectrum to stitch in L23 mode.')
        l2_xanes = self if self.edge == 'L2' else other
        l3_xanes = self if self.edge == 'L3' else other
        if l2_xanes.absorbing_element.number > 30:
            raise ValueError(f'Does not support L2,3-edge XANES for {l2_xanes.absorbing_element} element')
        l2_f = interp1d(l2_xanes.x, l2_xanes.y, bounds_error=False, fill_value='extrapolate', kind='cubic')
        l3_f = interp1d(l3_xanes.x, l3_xanes.y, bounds_error=True, fill_value=0, kind='cubic')
        energy = list(np.linspace(min(l3_xanes.x), max(l3_xanes.x), num=num_samples))
        mu = [i + j for i, j in zip([max(i, 0) for i in l2_f(energy)], l3_f(energy))]
        idx = energy.index(min(energy, key=lambda x: abs(x - l2_xanes.x[0])))
        if abs(mu[idx] - mu[idx - 1]) / mu[idx - 1] > 0.1:
            warnings.warn('There might exist a jump at the L2 and L3-edge junction.', UserWarning)
        return XAS(energy, mu, self.structure, self.absorbing_element, 'L23', 'XANES')
    raise ValueError('Invalid mode. Only XAFS and L23 are supported.')