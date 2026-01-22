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
def plot_zt_temp(self, doping='all', output: Literal['average', 'eigs']='average', relaxation_time=1e-14):
    """Plot the figure of merit zT in function of temperature for different doping levels.

        Args:
            doping (str): the default 'all' plots all the doping levels in the analyzer.
                Specify a list of doping levels if you want to plot only some.
            output: with 'average' you get an average of the three directions
                with 'eigs' you get all the three directions.
            relaxation_time: specify a constant relaxation time value

        Raises:
            ValueError: if output is not 'average' or 'eigs'

        Returns:
            a matplotlib object
        """
    if output not in ('average', 'eigs'):
        raise ValueError(f"output={output!r} must be 'average' or 'eigs'")
    zt = self._bz.get_zt(relaxation_time=relaxation_time, output=output)
    ax = pretty_plot(22, 14)
    tlist = sorted(zt['n'])
    doping = self._bz.doping['n'] if doping == 'all' else doping
    for idx, doping_type in enumerate(['n', 'p']):
        plt.subplot(121 + idx)
        for dop in doping:
            dop_idx = self._bz.doping[doping_type].index(dop)
            zt_temp = []
            for temp in tlist:
                zt_temp.append(zt[doping_type][temp][dop_idx])
            if output == 'average':
                ax.plot(tlist, zt_temp, marker='s', label=str(dop) + ' $cm^{-3}$')
            elif output == 'eigs':
                for xyz in range(3):
                    ax.plot(tlist, list(zip(*zt_temp))[xyz], marker='s', label=f'{xyz} {dop} $cm^{{-3}}$')
        ax.set_title(f'{doping_type}-type', fontsize=20)
        if idx == 0:
            ax.set_ylabel('zT', fontsize=30.0)
        ax.set_xlabel('Temperature (K)', fontsize=30.0)
        ax.legend(loc='best', fontsize=15)
        ax.grid()
        ax.set_xticks(fontsize=25)
        ax.set_yticks(fontsize=25)
    plt.tight_layout()
    return ax