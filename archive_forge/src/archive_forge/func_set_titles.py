from __future__ import annotations
import functools
import itertools
import warnings
from collections.abc import Hashable, Iterable, MutableMapping
from typing import TYPE_CHECKING, Any, Callable, Generic, Literal, TypeVar, cast
import numpy as np
from xarray.core.formatting import format_item
from xarray.core.types import HueStyleOptions, T_DataArrayOrSet
from xarray.plot.utils import (
def set_titles(self, template: str='{coord} = {value}', maxchar: int=30, size=None, **kwargs) -> None:
    """
        Draw titles either above each facet or on the grid margins.

        Parameters
        ----------
        template : str, default: "{coord} = {value}"
            Template for plot titles containing {coord} and {value}
        maxchar : int, default: 30
            Truncate titles at maxchar
        **kwargs : keyword args
            additional arguments to matplotlib.text

        Returns
        -------
        self: FacetGrid object

        """
    import matplotlib as mpl
    if size is None:
        size = mpl.rcParams['axes.labelsize']
    nicetitle = functools.partial(_nicetitle, maxchar=maxchar, template=template)
    if self._single_group:
        for d, ax in zip(self.name_dicts.flat, self.axs.flat):
            if d is not None:
                coord, value = list(d.items()).pop()
                title = nicetitle(coord, value, maxchar=maxchar)
                ax.set_title(title, size=size, **kwargs)
    else:
        for index, (ax, row_name, handle) in enumerate(zip(self.axs[:, -1], self.row_names, self.row_labels)):
            title = nicetitle(coord=self._row_var, value=row_name, maxchar=maxchar)
            if not handle:
                self.row_labels[index] = ax.annotate(title, xy=(1.02, 0.5), xycoords='axes fraction', rotation=270, ha='left', va='center', **kwargs)
            else:
                handle.set_text(title)
                handle.update(kwargs)
        for index, (ax, col_name, handle) in enumerate(zip(self.axs[0, :], self.col_names, self.col_labels)):
            title = nicetitle(coord=self._col_var, value=col_name, maxchar=maxchar)
            if not handle:
                self.col_labels[index] = ax.set_title(title, size=size, **kwargs)
            else:
                handle.set_text(title)
                handle.update(kwargs)