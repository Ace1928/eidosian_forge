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
def plot_path(line, lattice=None, coords_are_cartesian=False, ax: plt.Axes=None, **kwargs):
    """Adds a line passing through the coordinates listed in 'line' to a matplotlib Axes.

    Args:
        line: list of coordinates.
        lattice: Lattice object used to convert from reciprocal to Cartesian coordinates
        coords_are_cartesian: Set to True if you are providing
            coordinates in Cartesian coordinates. Defaults to False.
            Requires lattice if False.
        ax: matplotlib Axes or None if a new figure should be created.
        kwargs: kwargs passed to the matplotlib function 'plot'. Color defaults to red
            and linewidth to 3.

    Returns:
        matplotlib figure and matplotlib ax
    """
    ax, fig = get_ax3d_fig(ax)
    if 'color' not in kwargs:
        kwargs['color'] = 'r'
    if 'linewidth' not in kwargs:
        kwargs['linewidth'] = 3
    for k in range(1, len(line)):
        vertex1 = line[k - 1]
        vertex2 = line[k]
        if not coords_are_cartesian:
            if lattice is None:
                raise ValueError('coords_are_cartesian False requires the lattice')
            vertex1 = lattice.get_cartesian_coords(vertex1)
            vertex2 = lattice.get_cartesian_coords(vertex2)
        ax.plot(*zip(vertex1, vertex2), **kwargs)
    return (fig, ax)