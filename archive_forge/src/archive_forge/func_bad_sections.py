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
def bad_sections(self, key='wall_time', criterion='mean', nmax=5):
    """Return first `nmax` sections with worst value of key `key` using criterion `criterion`."""
    bad_sections = self._order_by_peff(key, criterion=criterion, reverse=False)
    return bad_sections[:nmax]