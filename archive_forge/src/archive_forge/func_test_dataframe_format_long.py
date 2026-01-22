from __future__ import annotations
from textwrap import dedent
import numpy as np
import pandas as pd
import pytest
import dask.array as da
import dask.dataframe as dd
from dask.dataframe.utils import get_string_dtype, pyarrow_strings_enabled
from dask.utils import maybe_pluralize
def test_dataframe_format_long():
    pytest.importorskip('jinja2')
    df = pd.DataFrame({'A': [1, 2, 3, 4, 5, 6, 7, 8] * 10, 'B': list('ABCDEFGH') * 10, 'C': pd.Categorical(list('AAABBBCC') * 10)})
    string_dtype = _format_string_dtype()
    footer = _format_footer()
    ddf = dd.from_pandas(df, 10)
    exp = dedent(f'        Dask DataFrame Structure:\n                            A       B                C\n        npartitions=10                                \n        0               int64  {string_dtype}  category[known]\n        8                 ...     ...              ...\n        ...               ...     ...              ...\n        72                ...     ...              ...\n        79                ...     ...              ...\n        {footer}')
    assert repr(ddf) == exp
    assert str(ddf) == exp
    exp = dedent(f'                            A       B                C\n        npartitions=10                                \n        0               int64  {string_dtype}  category[known]\n        8                 ...     ...              ...\n        ...               ...     ...              ...\n        72                ...     ...              ...\n        79                ...     ...              ...')
    assert ddf.to_string() == exp
    exp_table = f'<table border="1" class="dataframe">\n  <thead>\n    <tr style="text-align: right;">\n      <th></th>\n      <th>A</th>\n      <th>B</th>\n      <th>C</th>\n    </tr>\n    <tr>\n      <th>npartitions=10</th>\n      <th></th>\n      <th></th>\n      <th></th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>0</th>\n      <td>int64</td>\n      <td>{string_dtype}</td>\n      <td>category[known]</td>\n    </tr>\n    <tr>\n      <th>8</th>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n    </tr>\n    <tr>\n      <th>...</th>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n    </tr>\n    <tr>\n      <th>72</th>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n    </tr>\n    <tr>\n      <th>79</th>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n    </tr>\n  </tbody>\n</table>'
    exp = f'<div><strong>Dask DataFrame Structure:</strong></div>\n{exp_table}\n<div>{footer}</div>'
    assert ddf.to_html() == exp
    exp = f'<div><strong>Dask DataFrame Structure:</strong></div>\n<div>\n{style}{exp_table}\n</div>\n<div>{footer}</div>'
    assert ddf._repr_html_() == exp