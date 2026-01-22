from __future__ import annotations
import copy
import itertools
import random
import warnings
from typing import TYPE_CHECKING
import matplotlib.pyplot as plt
import numpy as np
from sympy import Symbol
from sympy.solvers import linsolve, solve
from pymatgen.analysis.wulff import WulffShape
from pymatgen.core import Structure
from pymatgen.core.composition import Composition
from pymatgen.core.surface import get_slab_regions
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.io.vasp.outputs import Locpot, Outcar
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.util.due import Doi, due
from pymatgen.util.plotting import pretty_plot
@staticmethod
def chempot_plot_addons(ax, xrange, ref_el, pad=2.4, rect=None, ylim=None):
    """
        Helper function to a chempot plot look nicer.

        Args:
            plt (Plot) Plot to add things to.
            xrange (list): xlim parameter
            ref_el (str): Element of the referenced chempot.
            axes(axes) Axes object from matplotlib
            pad (float) For tight layout
            rect (list): For tight layout
            ylim (ylim parameter):

        return (Plot): Modified plot with addons.
        return (Plot): Modified plot with addons.
        """
    plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.0)
    ax.set_xlabel(f'Chemical potential $\\Delta\\mu_{{{ref_el}}}$ (eV)')
    ylim = ylim or ax.get_ylim()
    plt.xticks(rotation=60)
    plt.ylim(ylim)
    xlim = ax.get_xlim()
    plt.xlim(xlim)
    plt.tight_layout(pad=pad, rect=rect or [-0.047, 0, 0.84, 1])
    plt.plot([xrange[0], xrange[0]], ylim, '--k')
    plt.plot([xrange[1], xrange[1]], ylim, '--k')
    xy = [np.mean([xrange[1]]), np.mean(ylim)]
    plt.annotate(f'{ref_el}-rich', xy=xy, xytext=xy, rotation=90, fontsize=17)
    xy = [np.mean([xlim[0]]), np.mean(ylim)]
    plt.annotate(f'{ref_el}-poor', xy=xy, xytext=xy, rotation=90, fontsize=17)
    return ax