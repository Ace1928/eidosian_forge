from __future__ import annotations
from contextlib import contextmanager
import copy
from functools import partial
import operator
from typing import (
import warnings
import numpy as np
from pandas._config import get_option
from pandas.compat._optional import import_optional_dependency
from pandas.util._decorators import (
from pandas.util._exceptions import find_stack_level
import pandas as pd
from pandas import (
import pandas.core.common as com
from pandas.core.frame import (
from pandas.core.generic import NDFrame
from pandas.core.shared_docs import _shared_docs
from pandas.io.formats.format import save_to_buffer
from pandas.io.formats.style_render import (
@doc(name='background', alt='text', image_prefix='bg', text_threshold='text_color_threshold : float or int\n\n            Luminance threshold for determining text color in [0, 1]. Facilitates text\n\n            visibility across varying background colors. All text is dark if 0, and\n\n            light if 1, defaults to 0.408.')
@Substitution(subset=subset_args)
def background_gradient(self, cmap: str | Colormap='PuBu', low: float=0, high: float=0, axis: Axis | None=0, subset: Subset | None=None, text_color_threshold: float=0.408, vmin: float | None=None, vmax: float | None=None, gmap: Sequence | None=None) -> Styler:
    """
        Color the {name} in a gradient style.

        The {name} color is determined according
        to the data in each column, row or frame, or by a given
        gradient map. Requires matplotlib.

        Parameters
        ----------
        cmap : str or colormap
            Matplotlib colormap.
        low : float
            Compress the color range at the low end. This is a multiple of the data
            range to extend below the minimum; good values usually in [0, 1],
            defaults to 0.
        high : float
            Compress the color range at the high end. This is a multiple of the data
            range to extend above the maximum; good values usually in [0, 1],
            defaults to 0.
        axis : {{0, 1, "index", "columns", None}}, default 0
            Apply to each column (``axis=0`` or ``'index'``), to each row
            (``axis=1`` or ``'columns'``), or to the entire DataFrame at once
            with ``axis=None``.
        %(subset)s
        {text_threshold}
        vmin : float, optional
            Minimum data value that corresponds to colormap minimum value.
            If not specified the minimum value of the data (or gmap) will be used.
        vmax : float, optional
            Maximum data value that corresponds to colormap maximum value.
            If not specified the maximum value of the data (or gmap) will be used.
        gmap : array-like, optional
            Gradient map for determining the {name} colors. If not supplied
            will use the underlying data from rows, columns or frame. If given as an
            ndarray or list-like must be an identical shape to the underlying data
            considering ``axis`` and ``subset``. If given as DataFrame or Series must
            have same index and column labels considering ``axis`` and ``subset``.
            If supplied, ``vmin`` and ``vmax`` should be given relative to this
            gradient map.

            .. versionadded:: 1.3.0

        Returns
        -------
        Styler

        See Also
        --------
        Styler.{alt}_gradient: Color the {alt} in a gradient style.

        Notes
        -----
        When using ``low`` and ``high`` the range
        of the gradient, given by the data if ``gmap`` is not given or by ``gmap``,
        is extended at the low end effectively by
        `map.min - low * map.range` and at the high end by
        `map.max + high * map.range` before the colors are normalized and determined.

        If combining with ``vmin`` and ``vmax`` the `map.min`, `map.max` and
        `map.range` are replaced by values according to the values derived from
        ``vmin`` and ``vmax``.

        This method will preselect numeric columns and ignore non-numeric columns
        unless a ``gmap`` is supplied in which case no preselection occurs.

        Examples
        --------
        >>> df = pd.DataFrame(columns=["City", "Temp (c)", "Rain (mm)", "Wind (m/s)"],
        ...                   data=[["Stockholm", 21.6, 5.0, 3.2],
        ...                         ["Oslo", 22.4, 13.3, 3.1],
        ...                         ["Copenhagen", 24.5, 0.0, 6.7]])

        Shading the values column-wise, with ``axis=0``, preselecting numeric columns

        >>> df.style.{name}_gradient(axis=0)  # doctest: +SKIP

        .. figure:: ../../_static/style/{image_prefix}_ax0.png

        Shading all values collectively using ``axis=None``

        >>> df.style.{name}_gradient(axis=None)  # doctest: +SKIP

        .. figure:: ../../_static/style/{image_prefix}_axNone.png

        Compress the color map from the both ``low`` and ``high`` ends

        >>> df.style.{name}_gradient(axis=None, low=0.75, high=1.0)  # doctest: +SKIP

        .. figure:: ../../_static/style/{image_prefix}_axNone_lowhigh.png

        Manually setting ``vmin`` and ``vmax`` gradient thresholds

        >>> df.style.{name}_gradient(axis=None, vmin=6.7, vmax=21.6)  # doctest: +SKIP

        .. figure:: ../../_static/style/{image_prefix}_axNone_vminvmax.png

        Setting a ``gmap`` and applying to all columns with another ``cmap``

        >>> df.style.{name}_gradient(axis=0, gmap=df['Temp (c)'], cmap='YlOrRd')
        ...  # doctest: +SKIP

        .. figure:: ../../_static/style/{image_prefix}_gmap.png

        Setting the gradient map for a dataframe (i.e. ``axis=None``), we need to
        explicitly state ``subset`` to match the ``gmap`` shape

        >>> gmap = np.array([[1,2,3], [2,3,4], [3,4,5]])
        >>> df.style.{name}_gradient(axis=None, gmap=gmap,
        ...     cmap='YlOrRd', subset=['Temp (c)', 'Rain (mm)', 'Wind (m/s)']
        ... )  # doctest: +SKIP

        .. figure:: ../../_static/style/{image_prefix}_axNone_gmap.png
        """
    if subset is None and gmap is None:
        subset = self._get_numeric_subset_default()
    self.apply(_background_gradient, cmap=cmap, subset=subset, axis=axis, low=low, high=high, text_color_threshold=text_color_threshold, vmin=vmin, vmax=vmax, gmap=gmap)
    return self