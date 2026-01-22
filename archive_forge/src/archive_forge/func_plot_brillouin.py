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
def plot_brillouin(self):
    """Plot the Brillouin zone.

        Returns:
            plt.Figure: A matplotlib figure object with the Brillouin zone.
        """
    labels = {}
    for k in self._bs[0].kpoints:
        if k.label:
            labels[k.label] = k.frac_coords
    lines = []
    for branch in self._bs[0].branches:
        kpts = self._bs[0].kpoints
        start_idx, end_idx = (branch['start_index'], branch['end_index'])
        lines.append([kpts[start_idx].frac_coords, kpts[end_idx].frac_coords])
    return plot_brillouin_zone(self._bs[0].lattice_rec, lines=lines, labels=labels)