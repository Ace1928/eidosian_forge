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
def set_tooltips(self, ttips: DataFrame, props: CSSProperties | None=None, css_class: str | None=None) -> Styler:
    """
        Set the DataFrame of strings on ``Styler`` generating ``:hover`` tooltips.

        These string based tooltips are only applicable to ``<td>`` HTML elements,
        and cannot be used for column or index headers.

        .. versionadded:: 1.3.0

        Parameters
        ----------
        ttips : DataFrame
            DataFrame containing strings that will be translated to tooltips, mapped
            by identical column and index values that must exist on the underlying
            Styler data. None, NaN values, and empty strings will be ignored and
            not affect the rendered HTML.
        props : list-like or str, optional
            List of (attr, value) tuples or a valid CSS string. If ``None`` adopts
            the internal default values described in notes.
        css_class : str, optional
            Name of the tooltip class used in CSS, should conform to HTML standards.
            Only useful if integrating tooltips with external CSS. If ``None`` uses the
            internal default value 'pd-t'.

        Returns
        -------
        Styler

        Notes
        -----
        Tooltips are created by adding `<span class="pd-t"></span>` to each data cell
        and then manipulating the table level CSS to attach pseudo hover and pseudo
        after selectors to produce the required the results.

        The default properties for the tooltip CSS class are:

        - visibility: hidden
        - position: absolute
        - z-index: 1
        - background-color: black
        - color: white
        - transform: translate(-20px, -20px)

        The property 'visibility: hidden;' is a key prerequisite to the hover
        functionality, and should always be included in any manual properties
        specification, using the ``props`` argument.

        Tooltips are not designed to be efficient, and can add large amounts of
        additional HTML for larger tables, since they also require that ``cell_ids``
        is forced to `True`.

        Examples
        --------
        Basic application

        >>> df = pd.DataFrame(data=[[0, 1], [2, 3]])
        >>> ttips = pd.DataFrame(
        ...    data=[["Min", ""], [np.nan, "Max"]], columns=df.columns, index=df.index
        ... )
        >>> s = df.style.set_tooltips(ttips).to_html()

        Optionally controlling the tooltip visual display

        >>> df.style.set_tooltips(ttips, css_class='tt-add', props=[
        ...     ('visibility', 'hidden'),
        ...     ('position', 'absolute'),
        ...     ('z-index', 1)])  # doctest: +SKIP
        >>> df.style.set_tooltips(ttips, css_class='tt-add',
        ...     props='visibility:hidden; position:absolute; z-index:1;')
        ... # doctest: +SKIP
        """
    if not self.cell_ids:
        raise NotImplementedError("Tooltips can only render with 'cell_ids' is True.")
    if not ttips.index.is_unique or not ttips.columns.is_unique:
        raise KeyError('Tooltips render only if `ttips` has unique index and columns.')
    if self.tooltips is None:
        self.tooltips = Tooltips()
    self.tooltips.tt_data = ttips
    if props:
        self.tooltips.class_properties = props
    if css_class:
        self.tooltips.class_name = css_class
    return self