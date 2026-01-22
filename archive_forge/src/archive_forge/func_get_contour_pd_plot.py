from __future__ import annotations
import collections
import itertools
import json
import logging
import math
import os
import re
import warnings
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Literal, no_type_check
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from matplotlib import cm
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.font_manager import FontProperties
from monty.json import MontyDecoder, MSONable
from scipy import interpolate
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from tqdm import tqdm
from pymatgen.analysis.reaction_calculator import Reaction, ReactionError
from pymatgen.core import DummySpecies, Element, get_el_sp
from pymatgen.core.composition import Composition
from pymatgen.entries import Entry
from pymatgen.util.coord import Simplex, in_coord_list
from pymatgen.util.due import Doi, due
from pymatgen.util.plotting import pretty_plot
from pymatgen.util.string import htmlify, latexify
def get_contour_pd_plot(self):
    """
        Plot a contour phase diagram plot, where phase triangles are colored
        according to degree of instability by interpolation. Currently only
        works for 3-component phase diagrams.

        Returns:
            A matplotlib plot object.
        """
    pd = self._pd
    entries = pd.qhull_entries
    data = np.array(pd.qhull_data)
    ax = self._get_matplotlib_2d_plot()
    data[:, 0:2] = triangular_coord(data[:, 0:2]).transpose()
    for idx, entry in enumerate(entries):
        data[idx, 2] = self._pd.get_e_above_hull(entry)
    gridsize = 0.005
    xnew = np.arange(0, 1.0, gridsize)
    ynew = np.arange(0, 1, gridsize)
    f = interpolate.LinearNDInterpolator(data[:, 0:2], data[:, 2])
    znew = np.zeros((len(ynew), len(xnew)))
    for idx, xval in enumerate(xnew):
        for j, yval in enumerate(ynew):
            znew[j, idx] = f(xval, yval)
    contourf = ax.contourf(xnew, ynew, znew, 1000, cmap=cm.autumn_r)
    plt.colorbar(contourf)
    return ax