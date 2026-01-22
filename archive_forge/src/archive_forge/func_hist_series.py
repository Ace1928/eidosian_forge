from __future__ import annotations
from typing import (
import numpy as np
from pandas.core.dtypes.common import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import (
from pandas.io.formats.printing import pprint_thing
from pandas.plotting._matplotlib.core import (
from pandas.plotting._matplotlib.groupby import (
from pandas.plotting._matplotlib.misc import unpack_single_str_list
from pandas.plotting._matplotlib.tools import (
def hist_series(self: Series, by=None, ax=None, grid: bool=True, xlabelsize: int | None=None, xrot=None, ylabelsize: int | None=None, yrot=None, figsize: tuple[float, float] | None=None, bins: int=10, legend: bool=False, **kwds):
    import matplotlib.pyplot as plt
    if legend and 'label' in kwds:
        raise ValueError('Cannot use both legend and label')
    if by is None:
        if kwds.get('layout', None) is not None:
            raise ValueError("The 'layout' keyword is not supported when 'by' is None")
        fig = kwds.pop('figure', plt.gcf() if plt.get_fignums() else plt.figure(figsize=figsize))
        if figsize is not None and tuple(figsize) != tuple(fig.get_size_inches()):
            fig.set_size_inches(*figsize, forward=True)
        if ax is None:
            ax = fig.gca()
        elif ax.get_figure() != fig:
            raise AssertionError('passed axis not bound to passed figure')
        values = self.dropna().values
        if legend:
            kwds['label'] = self.name
        ax.hist(values, bins=bins, **kwds)
        if legend:
            ax.legend()
        ax.grid(grid)
        axes = np.array([ax])
        set_ticks_props(axes, xlabelsize=xlabelsize, xrot=xrot, ylabelsize=ylabelsize, yrot=yrot)
    else:
        if 'figure' in kwds:
            raise ValueError("Cannot pass 'figure' when using the 'by' argument, since a new 'Figure' instance will be created")
        axes = _grouped_hist(self, by=by, ax=ax, grid=grid, figsize=figsize, bins=bins, xlabelsize=xlabelsize, xrot=xrot, ylabelsize=ylabelsize, yrot=yrot, legend=legend, **kwds)
    if hasattr(axes, 'ndim'):
        if axes.ndim == 1 and len(axes) == 1:
            return axes[0]
    return axes