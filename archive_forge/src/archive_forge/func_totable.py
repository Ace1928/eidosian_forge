from __future__ import annotations
import collections
import logging
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from pymatgen.io.core import ParseError
from pymatgen.util.plotting import add_fig_kwargs, get_ax_fig
def totable(self, stop=None, reverse=True):
    """
        Return table (list of lists) with timing results.

        Args:
            stop: Include results up to stop. None for all
            reverse: Put items with highest wall_time in first positions if True.
        """
    osects = self._order_by_peff('wall_time', criterion='mean', reverse=reverse)
    if stop is not None:
        osects = osects[:stop]
    n = len(self.filenames)
    table = [['AbinitTimerSection', *alternate(self.filenames, n * ['%'])]]
    for sect_name in osects:
        peff = self[sect_name]['wall_time']
        fract = self[sect_name]['wall_fract']
        vals = alternate(peff, fract)
        table.append([sect_name] + [f'{val:.2f}' for val in vals])
    return table