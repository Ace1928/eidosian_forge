from __future__ import annotations
import collections
import itertools
from math import acos, pi
from typing import TYPE_CHECKING
from warnings import warn
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Voronoi
from pymatgen.analysis.local_env import JmolNN, VoronoiNN
from pymatgen.core import Composition, Element, PeriodicSite, Species
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
@staticmethod
def plot_vor_analysis(voronoi_ensemble: list[tuple[str, float]]) -> plt.Axes:
    """Plot the Voronoi analysis.

        Args:
            voronoi_ensemble (list[tuple[str, float]]): List of tuples containing labels and
                values for Voronoi analysis.

        Returns:
            plt.Axes: Matplotlib Axes object with the plotted Voronoi analysis.
        """
    labels, val = zip(*voronoi_ensemble)
    arr = np.array(val, dtype=float)
    arr /= np.sum(arr)
    pos = np.arange(len(arr)) + 0.5
    _fig, ax = plt.subplots()
    ax.barh(pos, arr, align='center', alpha=0.5)
    ax.set_yticks(pos)
    ax.set_yticklabels(labels)
    ax.set(title='Voronoi Spectra', xlabel='Count')
    ax.grid(visible=True)
    return ax