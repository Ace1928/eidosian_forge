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
def get_tick_params(self, which='major'):
    """
        Get appearance parameters for ticks, ticklabels, and gridlines.

        .. versionadded:: 3.7

        Parameters
        ----------
        which : {'major', 'minor'}, default: 'major'
            The group of ticks for which the parameters are retrieved.

        Returns
        -------
        dict
            Properties for styling tick elements added to the axis.

        Notes
        -----
        This method returns the appearance parameters for styling *new*
        elements added to this axis and may be different from the values
        on current elements if they were modified directly by the user
        (e.g., via ``set_*`` methods on individual tick objects).

        Examples
        --------
        ::

            >>> ax.yaxis.set_tick_params(labelsize=30, labelcolor='red',
                                         direction='out', which='major')
            >>> ax.yaxis.get_tick_params(which='major')
            {'direction': 'out',
            'left': True,
            'right': False,
            'labelleft': True,
            'labelright': False,
            'gridOn': False,
            'labelsize': 30,
            'labelcolor': 'red'}
            >>> ax.yaxis.get_tick_params(which='minor')
            {'left': True,
            'right': False,
            'labelleft': True,
            'labelright': False,
            'gridOn': False}


        """
    _api.check_in_list(['major', 'minor'], which=which)
    if which == 'major':
        return self._translate_tick_params(self._major_tick_kw, reverse=True)
    return self._translate_tick_params(self._minor_tick_kw, reverse=True)