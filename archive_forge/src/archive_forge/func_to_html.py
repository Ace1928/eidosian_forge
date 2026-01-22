from __future__ import annotations
import collections
from collections import abc
from collections.abc import (
import functools
from inspect import signature
from io import StringIO
import itertools
import operator
import sys
from textwrap import dedent
from typing import (
import warnings
import numpy as np
from numpy import ma
from pandas._config import (
from pandas._config.config import _get_option
from pandas._libs import (
from pandas._libs.hashtable import duplicated
from pandas._libs.lib import is_range_indexer
from pandas.compat import PYPY
from pandas.compat._constants import REF_COUNT
from pandas.compat._optional import import_optional_dependency
from pandas.compat.numpy import function as nv
from pandas.errors import (
from pandas.util._decorators import (
from pandas.util._exceptions import (
from pandas.util._validators import (
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.concat import concat_compat
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.missing import (
from pandas.core import (
from pandas.core.accessor import CachedAccessor
from pandas.core.apply import reconstruct_and_relabel_result
from pandas.core.array_algos.take import take_2d_multi
from pandas.core.arraylike import OpsMixin
from pandas.core.arrays import (
from pandas.core.arrays.sparse import SparseFrameAccessor
from pandas.core.construction import (
from pandas.core.generic import (
from pandas.core.indexers import check_key_length
from pandas.core.indexes.api import (
from pandas.core.indexes.multi import (
from pandas.core.indexing import (
from pandas.core.internals import (
from pandas.core.internals.construction import (
from pandas.core.methods import selectn
from pandas.core.reshape.melt import melt
from pandas.core.series import Series
from pandas.core.shared_docs import _shared_docs
from pandas.core.sorting import (
from pandas.io.common import get_handle
from pandas.io.formats import (
from pandas.io.formats.info import (
import pandas.plotting
@deprecate_nonkeyword_arguments(version='3.0', allowed_args=['self', 'buf'], name='to_html')
@Substitution(header_type='bool', header='Whether to print column labels, default True', col_space_type='str or int, list or dict of int or str', col_space='The minimum width of each column in CSS length units.  An int is assumed to be px units.')
@Substitution(shared_params=fmt.common_docstring, returns=fmt.return_docstring)
def to_html(self, buf: FilePath | WriteBuffer[str] | None=None, columns: Axes | None=None, col_space: ColspaceArgType | None=None, header: bool=True, index: bool=True, na_rep: str='NaN', formatters: FormattersType | None=None, float_format: FloatFormatType | None=None, sparsify: bool | None=None, index_names: bool=True, justify: str | None=None, max_rows: int | None=None, max_cols: int | None=None, show_dimensions: bool | str=False, decimal: str='.', bold_rows: bool=True, classes: str | list | tuple | None=None, escape: bool=True, notebook: bool=False, border: int | bool | None=None, table_id: str | None=None, render_links: bool=False, encoding: str | None=None) -> str | None:
    """
        Render a DataFrame as an HTML table.
        %(shared_params)s
        bold_rows : bool, default True
            Make the row labels bold in the output.
        classes : str or list or tuple, default None
            CSS class(es) to apply to the resulting html table.
        escape : bool, default True
            Convert the characters <, >, and & to HTML-safe sequences.
        notebook : {True, False}, default False
            Whether the generated HTML is for IPython Notebook.
        border : int
            A ``border=border`` attribute is included in the opening
            `<table>` tag. Default ``pd.options.display.html.border``.
        table_id : str, optional
            A css id is included in the opening `<table>` tag if specified.
        render_links : bool, default False
            Convert URLs to HTML links.
        encoding : str, default "utf-8"
            Set character encoding.
        %(returns)s
        See Also
        --------
        to_string : Convert DataFrame to a string.

        Examples
        --------
        >>> df = pd.DataFrame(data={'col1': [1, 2], 'col2': [4, 3]})
        >>> html_string = '''<table border="1" class="dataframe">
        ...   <thead>
        ...     <tr style="text-align: right;">
        ...       <th></th>
        ...       <th>col1</th>
        ...       <th>col2</th>
        ...     </tr>
        ...   </thead>
        ...   <tbody>
        ...     <tr>
        ...       <th>0</th>
        ...       <td>1</td>
        ...       <td>4</td>
        ...     </tr>
        ...     <tr>
        ...       <th>1</th>
        ...       <td>2</td>
        ...       <td>3</td>
        ...     </tr>
        ...   </tbody>
        ... </table>'''
        >>> assert html_string == df.to_html()
        """
    if justify is not None and justify not in fmt.VALID_JUSTIFY_PARAMETERS:
        raise ValueError('Invalid value for justify parameter')
    formatter = fmt.DataFrameFormatter(self, columns=columns, col_space=col_space, na_rep=na_rep, header=header, index=index, formatters=formatters, float_format=float_format, bold_rows=bold_rows, sparsify=sparsify, justify=justify, index_names=index_names, escape=escape, decimal=decimal, max_rows=max_rows, max_cols=max_cols, show_dimensions=show_dimensions)
    return fmt.DataFrameRenderer(formatter).to_html(buf=buf, classes=classes, notebook=notebook, border=border, encoding=encoding, table_id=table_id, render_links=render_links)