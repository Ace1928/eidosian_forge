import itertools
import logging
import locale
import math
from numbers import Integral
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook
from matplotlib import transforms as mtransforms
def format_ticks(self, values):
    """Return the tick labels for all the ticks at once."""
    self.set_locs(values)
    return [self(value, i) for i, value in enumerate(values)]