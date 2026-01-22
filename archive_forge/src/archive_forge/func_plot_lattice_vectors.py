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
def plot_lattice_vectors(lattice, ax: plt.Axes=None, **kwargs):
    """Adds the basis vectors of the lattice provided to a matplotlib Axes.

    Args:
        lattice: Lattice object
        ax: matplotlib Axes or None if a new figure should be created.
        kwargs: kwargs passed to the matplotlib function 'plot'. Color defaults to green
            and linewidth to 3.

    Returns:
        matplotlib figure and matplotlib ax
    """
    ax, fig = get_ax3d_fig(ax)
    if 'color' not in kwargs:
        kwargs['color'] = 'g'
    if 'linewidth' not in kwargs:
        kwargs['linewidth'] = 3
    vertex1 = lattice.get_cartesian_coords([0.0, 0.0, 0.0])
    vertex2 = lattice.get_cartesian_coords([1.0, 0.0, 0.0])
    ax.plot(*zip(vertex1, vertex2), **kwargs)
    vertex2 = lattice.get_cartesian_coords([0.0, 1.0, 0.0])
    ax.plot(*zip(vertex1, vertex2), **kwargs)
    vertex2 = lattice.get_cartesian_coords([0.0, 0.0, 1.0])
    ax.plot(*zip(vertex1, vertex2), **kwargs)
    return (fig, ax)