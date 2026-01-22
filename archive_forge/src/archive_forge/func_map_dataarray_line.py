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
def map_dataarray_line(self: T_FacetGrid, func: Callable, x: Hashable | None, y: Hashable | None, hue: Hashable | None, add_legend: bool=True, _labels=None, **kwargs: Any) -> T_FacetGrid:
    from xarray.plot.dataarray_plot import _infer_line_data
    for d, ax in zip(self.name_dicts.flat, self.axs.flat):
        if d is not None:
            subset = self.data.loc[d]
            mappable = func(subset, x=x, y=y, ax=ax, hue=hue, add_legend=False, _labels=False, **kwargs)
            self._mappables.append(mappable)
    xplt, yplt, hueplt, huelabel = _infer_line_data(darray=self.data.loc[self.name_dicts.flat[0]], x=x, y=y, hue=hue)
    xlabel = label_from_attrs(xplt)
    ylabel = label_from_attrs(yplt)
    self._hue_var = hueplt
    self._finalize_grid(xlabel, ylabel)
    if add_legend and hueplt is not None and (huelabel is not None):
        self.add_legend(label=huelabel)
    return self