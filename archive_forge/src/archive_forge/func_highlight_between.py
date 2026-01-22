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
@Substitution(subset=subset_args, color=coloring_args.format(default='yellow'), props=properties_args)
def highlight_between(self, subset: Subset | None=None, color: str='yellow', axis: Axis | None=0, left: Scalar | Sequence | None=None, right: Scalar | Sequence | None=None, inclusive: IntervalClosedType='both', props: str | None=None) -> Styler:
    """
        Highlight a defined range with a style.

        .. versionadded:: 1.3.0

        Parameters
        ----------
        %(subset)s
        %(color)s
        axis : {0 or 'index', 1 or 'columns', None}, default 0
            If ``left`` or ``right`` given as sequence, axis along which to apply those
            boundaries. See examples.
        left : scalar or datetime-like, or sequence or array-like, default None
            Left bound for defining the range.
        right : scalar or datetime-like, or sequence or array-like, default None
            Right bound for defining the range.
        inclusive : {'both', 'neither', 'left', 'right'}
            Identify whether bounds are closed or open.
        %(props)s

        Returns
        -------
        Styler

        See Also
        --------
        Styler.highlight_null: Highlight missing values with a style.
        Styler.highlight_max: Highlight the maximum with a style.
        Styler.highlight_min: Highlight the minimum with a style.
        Styler.highlight_quantile: Highlight values defined by a quantile with a style.

        Notes
        -----
        If ``left`` is ``None`` only the right bound is applied.
        If ``right`` is ``None`` only the left bound is applied. If both are ``None``
        all values are highlighted.

        ``axis`` is only needed if ``left`` or ``right`` are provided as a sequence or
        an array-like object for aligning the shapes. If ``left`` and ``right`` are
        both scalars then all ``axis`` inputs will give the same result.

        This function only works with compatible ``dtypes``. For example a datetime-like
        region can only use equivalent datetime-like ``left`` and ``right`` arguments.
        Use ``subset`` to control regions which have multiple ``dtypes``.

        Examples
        --------
        Basic usage

        >>> df = pd.DataFrame({
        ...     'One': [1.2, 1.6, 1.5],
        ...     'Two': [2.9, 2.1, 2.5],
        ...     'Three': [3.1, 3.2, 3.8],
        ... })
        >>> df.style.highlight_between(left=2.1, right=2.9)  # doctest: +SKIP

        .. figure:: ../../_static/style/hbetw_basic.png

        Using a range input sequence along an ``axis``, in this case setting a ``left``
        and ``right`` for each column individually

        >>> df.style.highlight_between(left=[1.4, 2.4, 3.4], right=[1.6, 2.6, 3.6],
        ...     axis=1, color="#fffd75")  # doctest: +SKIP

        .. figure:: ../../_static/style/hbetw_seq.png

        Using ``axis=None`` and providing the ``left`` argument as an array that
        matches the input DataFrame, with a constant ``right``

        >>> df.style.highlight_between(left=[[2,2,3],[2,2,3],[3,3,3]], right=3.5,
        ...     axis=None, color="#fffd75")  # doctest: +SKIP

        .. figure:: ../../_static/style/hbetw_axNone.png

        Using ``props`` instead of default background coloring

        >>> df.style.highlight_between(left=1.5, right=3.5,
        ...     props='font-weight:bold;color:#e83e8c')  # doctest: +SKIP

        .. figure:: ../../_static/style/hbetw_props.png
        """
    if props is None:
        props = f'background-color: {color};'
    return self.apply(_highlight_between, axis=axis, subset=subset, props=props, left=left, right=right, inclusive=inclusive)