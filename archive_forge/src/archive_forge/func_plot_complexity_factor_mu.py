from __future__ import annotations
import copy
import itertools
import logging
import math
import typing
import warnings
from collections import Counter
from typing import TYPE_CHECKING, Literal, cast, no_type_check
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import palettable
import scipy.interpolate as scint
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec
from monty.dev import requires
from monty.json import jsanitize
from pymatgen.core import Element
from pymatgen.electronic_structure.bandstructure import BandStructureSymmLine
from pymatgen.electronic_structure.boltztrap import BoltztrapError
from pymatgen.electronic_structure.core import OrbitalType, Spin
from pymatgen.util.plotting import add_fig_kwargs, get_ax3d_fig, pretty_plot
def plot_complexity_factor_mu(self, temps=(300,), output='average', Lambda=0.5):
    """Plot respect to the chemical potential of the Fermi surface complexity
        factor calculated as explained in Ref.
        Gibbs, Z. M. et al., Effective mass and fermi surface complexity factor
        from ab initio band structure calculations.
        npj Computational Materials 3, 8 (2017).

        Args:
            output: 'average' returns the complexity factor calculated using the average
                of the three diagonal components of the seebeck and conductivity tensors.
                'tensor' returns the complexity factor respect to the three
                diagonal components of seebeck and conductivity tensors.
            temps: list of temperatures of calculated seebeck and conductivity.
            Lambda: fitting parameter used to model the scattering (0.5 means constant
                relaxation time).

        Returns:
            a matplotlib object
        """
    ax = pretty_plot(9, 7)
    for T in temps:
        cmplx_fact = self._bz.get_complexity_factor(output=output, temp=T, Lambda=Lambda)
        start = self._bz.mu_doping['p'][T][0]
        stop = self._bz.mu_doping['n'][T][0]
        mu_steps_1 = []
        mu_steps_2 = []
        cmplx_fact_1 = []
        cmplx_fact_2 = []
        for i, mu in enumerate(self._bz.mu_steps):
            if mu <= start:
                mu_steps_1.append(mu)
                cmplx_fact_1.append(cmplx_fact[i])
            elif mu >= stop:
                mu_steps_2.append(mu)
                cmplx_fact_2.append(cmplx_fact[i])
        ax.plot(mu_steps_1, cmplx_fact_1, label=str(T) + 'K', linewidth=3.0)
        ax.plot(mu_steps_2, cmplx_fact_2, linewidth=3.0)
        if output == 'average':
            ax.gca().get_lines()[1].set_c(ax.gca().get_lines()[0].get_c())
        elif output == 'tensor':
            ax.gca().get_lines()[3].set_c(ax.gca().get_lines()[0].get_c())
            ax.gca().get_lines()[4].set_c(ax.gca().get_lines()[1].get_c())
            ax.gca().get_lines()[5].set_c(ax.gca().get_lines()[2].get_c())
    ax.set_xlabel('E-E$_f$ (eV)', fontsize=30)
    ax.set_ylabel('Complexity Factor', fontsize=30)
    ax.set_xticks(fontsize=25)
    ax.set_yticks(fontsize=25)
    if output == 'tensor':
        ax.legend([f'{dim}_{T}K' for T in temps for dim in ('x', 'y', 'z')], fontsize=20)
    elif output == 'average':
        ax.legend(fontsize=20)
    plt.tight_layout()
    return ax