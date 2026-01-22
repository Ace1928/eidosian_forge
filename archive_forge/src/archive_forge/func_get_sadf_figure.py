from __future__ import annotations
import logging
import time
from typing import TYPE_CHECKING
import matplotlib.pyplot as plt
import numpy as np
from monty.json import MSONable
from scipy.spatial import Voronoi
from pymatgen.analysis.chemenv.utils.coordination_geometry_utils import (
from pymatgen.analysis.chemenv.utils.defs_utils import AdditionalConditions
from pymatgen.analysis.chemenv.utils.math_utils import normal_cdf_step
from pymatgen.core.sites import PeriodicSite
from pymatgen.core.structure import Structure
def get_sadf_figure(self, isite, normalized=True, figsize=None, step_function=None):
    """
        Get the Solid Angle Distribution Figure for a given site.

        Args:
            isite: Index of the site.
            normalized: Whether to normalize angles.
            figsize: Size of the figure.
            step_function: Type of step function to be used for the SADF.

        Returns:
            plt.figure: matplotlib figure.
        """

    def ap_func(ap):
        return np.power(ap, -0.1)
    if step_function is None:
        step_function = {'type': 'step_function', 'scale': 0.0001}
    fig = plt.figure() if figsize is None else plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    angs = self.neighbors_normalized_angles[isite] if normalized else self.neighbors_angles[isite]
    if step_function['type'] == 'step_function':
        isorted = np.argsort([ap_func(aa['min']) for aa in angs])
        sorted_angs = [ap_func(angs[ii]['min']) for ii in isorted]
        dnb_angs = [len(angs[ii]['dnb_indices']) for ii in isorted]
        xx = [0.0]
        yy = [0.0]
        for iang, ang in enumerate(sorted_angs):
            xx.extend((ang, ang))
            yy.extend((yy[-1], yy[-1] + dnb_angs[iang]))
        xx.append(1.1 * xx[-1])
        yy.append(yy[-1])
    elif step_function['type'] == 'normal_cdf':
        scale = step_function['scale']
        _angles = [ap_func(aa['min']) for aa in angs]
        _dcns = [len(dd['dnb_indices']) for dd in angs]
        xx = np.linspace(0.0, 1.1 * max(_angles), num=500)
        yy = np.zeros_like(xx)
        for iang, ang in enumerate(_angles):
            yy += _dcns[iang] * normal_cdf_step(xx, mean=ang, scale=scale)
    else:
        raise ValueError(f'Step function of type {step_function['type']!r} is not allowed')
    ax.plot(xx, yy)
    return fig