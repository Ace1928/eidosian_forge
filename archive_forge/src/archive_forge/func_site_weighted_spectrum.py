from __future__ import annotations
import math
import warnings
from typing import Literal
import numpy as np
from scipy.interpolate import interp1d
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.spectrum import Spectrum
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
def site_weighted_spectrum(xas_list: list[XAS], num_samples: int=500) -> XAS:
    """
    Obtain site-weighted XAS object based on site multiplicity for each
    absorbing index and its corresponding site-wise spectrum.

    Args:
        xas_list([XAS]): List of XAS object to be weighted
        num_samples(int): Number of samples for interpolation

    Returns:
        XAS object: The site-weighted spectrum
    """
    matcher = StructureMatcher()
    groups = matcher.group_structures([i.structure for i in xas_list])
    if len(groups) > 1:
        raise ValueError('The input structures mismatch')
    if not len({i.absorbing_element for i in xas_list}) == len({i.edge for i in xas_list}) == 1:
        raise ValueError('Can only perform site-weighting for spectra with same absorbing element and same absorbing edge.')
    if len({i.absorbing_index for i in xas_list}) == 1 or None in {i.absorbing_index for i in xas_list}:
        raise ValueError('Need at least two site-wise spectra to perform site-weighting')
    sa = SpacegroupAnalyzer(groups[0][0])
    ss = sa.get_symmetrized_structure()
    maxes, mines = ([], [])
    fs = []
    multiplicities = []
    for xas in xas_list:
        multiplicity = len(ss.find_equivalent_sites(ss[xas.absorbing_index]))
        multiplicities.append(multiplicity)
        maxes.append(max(xas.x))
        mines.append(min(xas.x))
        f = interp1d(np.asarray(xas.x), np.asarray(xas.y), bounds_error=False, fill_value=0, kind='cubic')
        fs.append(f)
    x_axis = np.linspace(max(mines), min(maxes), num=num_samples)
    weighted_spectrum = np.zeros(num_samples)
    sum_multiplicities = sum(multiplicities)
    for i, j in enumerate(multiplicities):
        weighted_spectrum += j * fs[i](x_axis) / sum_multiplicities
    return XAS(x_axis, weighted_spectrum, ss, xas.absorbing_element, xas.edge, xas.spectrum_type)