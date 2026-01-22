from __future__ import annotations
import logging
from collections import namedtuple
from typing import TYPE_CHECKING, Callable
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const
from matplotlib.collections import LineCollection
from monty.json import jsanitize
from pymatgen.electronic_structure.plotter import BSDOSPlotter, plot_brillouin_zone
from pymatgen.phonon.bandstructure import PhononBandStructureSymmLine
from pymatgen.phonon.gruneisen import GruneisenPhononBandStructureSymmLine
from pymatgen.util.plotting import add_fig_kwargs, get_ax_fig, pretty_plot
def get_plot_gs(self, ylim: float | None=None, **kwargs) -> Axes:
    """Get a matplotlib object for the Gruneisen bandstructure plot.

        Args:
            ylim: Specify the y-axis (gruneisen) limits; by default None let
                the code choose.
            **kwargs: additional keywords passed to ax.plot().
        """
    ax = pretty_plot(12, 8)
    kwargs.setdefault('linewidth', 2)
    kwargs.setdefault('marker', 'o')
    kwargs.setdefault('markersize', 2)
    data = self.bs_plot_data()
    for dist_idx in range(len(data['distances'])):
        for band_idx in range(self.n_bands):
            ys = [data['gruneisen'][dist_idx][band_idx][idx] for idx in range(len(data['distances'][dist_idx]))]
            ax.plot(data['distances'][dist_idx], ys, 'b-', **kwargs)
    self._make_ticks(ax)
    ax.axhline(0, linewidth=1, color='black')
    ax.set_xlabel('$\\mathrm{Wave\\ Vector}$', fontsize=30)
    ax.set_ylabel('$\\mathrm{Grüneisen\\ Parameter}$', fontsize=30)
    x_max = data['distances'][-1][-1]
    ax.set_xlim(0, x_max)
    if ylim is not None:
        ax.set_ylim(ylim)
    plt.tight_layout()
    return ax