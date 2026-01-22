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
def plot_eff_mass_temp(self, doping='all', output: Literal['average', 'eigs']='average'):
    """Plot the average effective mass in function of temperature
        for different doping levels.

        Args:
            doping (str): the default 'all' plots all the doping levels in the analyzer.
                Specify a list of doping levels if you want to plot only some.
            output ('average' | 'eigs'): with 'average' you get an average of the three directions
                with 'eigs' you get all the three directions.

        Returns:
            a matplotlib Axes object
        """
    if output == 'average':
        eff_mass = self._bz.get_average_eff_mass(output='average')
    elif output == 'eigs':
        eff_mass = self._bz.get_average_eff_mass(output='eigs')
    ax_main = pretty_plot(22, 14)
    tlist = sorted(eff_mass['n'])
    doping = self._bz.doping['n'] if doping == 'all' else doping
    for idx, doping_type in enumerate(['n', 'p']):
        ax = plt.subplot(121 + idx)
        for dop in doping:
            dop_idx = self._bz.doping[doping_type].index(dop)
            em_temp = [eff_mass[doping_type][temp][dop_idx] for temp in tlist]
            if output == 'average':
                ax.plot(tlist, em_temp, marker='s', label=f'{dop} $cm^{{-3}}$')
            elif output == 'eigs':
                for xyz in range(3):
                    ax.plot(tlist, list(zip(*em_temp))[xyz], marker='s', label=f'{xyz} {dop} $cm^{{-3}}$')
        ax.set_title(f'{doping_type}-type', fontsize=20)
        if idx == 0:
            ax.set_ylabel('Effective mass (m$_e$)', fontsize=30.0)
        ax.set_xlabel('Temperature (K)', fontsize=30.0)
        ax.legend(loc='best', fontsize=15)
        ax.grid()
        ax.tick_params(labelsize=25)
    plt.tight_layout()
    return ax_main