import datetime
import functools
import logging
from numbers import Real
import warnings
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook
import matplotlib.artist as martist
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.scale as mscale
import matplotlib.text as mtext
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
import matplotlib.units as munits
def get_ticklabels(self, minor=False, which=None):
    """
        Get this Axis' tick labels.

        Parameters
        ----------
        minor : bool
           Whether to return the minor or the major ticklabels.

        which : None, ('minor', 'major', 'both')
           Overrides *minor*.

           Selects which ticklabels to return

        Returns
        -------
        list of `~matplotlib.text.Text`
        """
    if which is not None:
        if which == 'minor':
            return self.get_minorticklabels()
        elif which == 'major':
            return self.get_majorticklabels()
        elif which == 'both':
            return self.get_majorticklabels() + self.get_minorticklabels()
        else:
            _api.check_in_list(['major', 'minor', 'both'], which=which)
    if minor:
        return self.get_minorticklabels()
    return self.get_majorticklabels()