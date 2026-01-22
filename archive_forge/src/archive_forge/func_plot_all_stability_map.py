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
def plot_all_stability_map(self, max_r, increments=50, delu_dict=None, delu_default=0, ax=None, labels=None, from_sphere_area=False, e_units='keV', r_units='nanometers', normalize=False, scale_per_atom=False):
    """
        Returns the plot of the formation energy of a particles
            of different polymorphs against its effect radius.

        Args:
            max_r (float): The maximum radius of the particle to plot up to.
            increments (int): Number of plot points
            delu_dict (dict): Dictionary of the chemical potentials to be set as
                constant. Note the key should be a sympy Symbol object of the
                format: Symbol("delu_el") where el is the name of the element.
            delu_default (float): Default value for all unset chemical potentials
            plt (pyplot): Plot
            labels (list): List of labels for each plot, corresponds to the
                list of se_analyzers
            from_sphere_area (bool): There are two ways to calculate the bulk
                formation energy. Either by treating the volume and thus surface
                area of the particle as a perfect sphere, or as a Wulff shape.

        Returns:
            plt.Axes: matplotlib Axes object
        """
    ax = ax or pretty_plot(width=8, height=7)
    for idx, analyzer in enumerate(self.se_analyzers):
        label = labels[idx] if labels else ''
        ax = self.plot_one_stability_map(analyzer, max_r, delu_dict, label=label, ax=ax, increments=increments, delu_default=delu_default, from_sphere_area=from_sphere_area, e_units=e_units, r_units=r_units, normalize=normalize, scale_per_atom=scale_per_atom)
    return ax