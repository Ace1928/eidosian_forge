from __future__ import annotations
import abc
import collections
from typing import TYPE_CHECKING
import matplotlib.pyplot as plt
import numpy as np
from pymatgen.core.spectrum import Spectrum
from pymatgen.util.plotting import add_fig_kwargs, pretty_plot
@add_fig_kwargs
def plot_structures(self, structures, fontsize=6, **kwargs):
    """
        Plot diffraction patterns for multiple structures on the same figure.

        Args:
            structures (Structure): List of structures
            two_theta_range ([float of length 2]): Tuple for range of
                two_thetas to calculate in degrees. Defaults to (0, 90). Set to
                None if you want all diffracted beams within the limiting
                sphere of radius 2 / wavelength.
            annotate_peaks (str | None): Whether and how to annotate the peaks
                with hkl indices. Default is 'compact', i.e. show short
                version (oriented vertically), e.g. 100. If 'full', show
                long version, e.g. (1, 0, 0). If None, do not show anything.
            fontsize: (int) fontsize for peak labels.
        """
    n_rows = len(structures)
    fig, axes = plt.subplots(nrows=n_rows, ncols=1, sharex=True, squeeze=False)
    for i, (ax, structure) in enumerate(zip(axes.ravel(), structures)):
        self.get_plot(structure, fontsize=fontsize, ax=ax, with_labels=i == n_rows - 1, **kwargs)
        spg_symbol, spg_number = structure.get_space_group_info()
        ax.set_title(f'{structure.formula} {spg_symbol} ({spg_number}) ')
    return fig