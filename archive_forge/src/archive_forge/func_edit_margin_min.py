import itertools
import kiwisolver as kiwi
import logging
import numpy as np
import matplotlib as mpl
import matplotlib.patches as mpatches
from matplotlib.transforms import Bbox
def edit_margin_min(self, todo, size, cell=0):
    """
        Change the minimum size of the margin for one cell.

        Parameters
        ----------
        todo : string (one of 'left', 'right', 'bottom', 'top')
            margin to alter.

        size : float
            Minimum size of the margin .  If it is larger than the
            existing minimum it updates the margin size. Fraction of
            figure size.

        cell : int
            Cell column or row to edit.
        """
    if size > self.margin_vals[todo][cell]:
        self.edit_margin(todo, size, cell)