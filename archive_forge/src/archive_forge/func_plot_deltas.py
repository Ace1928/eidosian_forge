from abc import ABCMeta, abstractmethod
import warnings
from typing import Any, Dict, Sequence, Tuple, TypeVar, Union
import numpy as np
from ase.utils.plotting import SimplePlottingAxes
def plot_deltas(self, ax: 'matplotlib.axes.Axes'=None, show: bool=False, filename: str=None, mplargs: dict=None) -> 'matplotlib.axes.Axes':
    """Simple plot of sparse DOS data as a set of delta functions

        Items at the same x-value can overlap and will not be summed together

        Args:
            ax: existing Matplotlib axes object. If not provided, a new figure
                with one set of axes will be created using Pyplot
            show: show the figure on-screen
            filename: if a path is given, save the figure to this file
            mplargs: additional arguments to pass to matplotlib Axes.vlines
                command (e.g. {'linewidth': 2} for a thicker line).

        Returns:
            Plotting axes. If "ax" was set, this is the same object.
        """
    if mplargs is None:
        mplargs = {}
    with SimplePlottingAxes(ax=ax, show=show, filename=filename) as ax:
        ax.vlines(self.get_energies(), 0, self.get_weights(), **mplargs)
    return ax