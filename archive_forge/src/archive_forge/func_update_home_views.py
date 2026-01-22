import enum
import functools
import re
import time
from types import SimpleNamespace
import uuid
from weakref import WeakKeyDictionary
import numpy as np
import matplotlib as mpl
from matplotlib._pylab_helpers import Gcf
from matplotlib import _api, cbook
def update_home_views(self, figure=None):
    """
        Make sure that ``self.home_views`` has an entry for all axes present
        in the figure.
        """
    if not figure:
        figure = self.figure
    for a in figure.get_axes():
        if a not in self.home_views[figure]:
            self.home_views[figure][a] = a._get_view()