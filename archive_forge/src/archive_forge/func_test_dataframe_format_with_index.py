from __future__ import annotations
from textwrap import dedent
import numpy as np
import pandas as pd
import pytest
import dask.array as da
import dask.dataframe as dd
from dask.dataframe.utils import get_string_dtype, pyarrow_strings_enabled
from dask.utils import maybe_pluralize
def test_dataframe_format_with_index():
    pytest.importorskip('jinja2')
    df = pd.DataFrame({'A': [1, 2, 3, 4, 5, 6, 7, 8], 'B': list('ABCDEFGH'), 'C': pd.Categorical(list('AAABBBCC'))}, index=list('ABCDEFGH'))
    ddf = dd.from_pandas(df, 3)
    string_dtype = _format_string_dtype()
    footer = _format_footer()
    exp = dedent(f'        Dask DataFrame Structure:\n                           A       B                C\n        npartitions=3                                \n        A              int64  {string_dtype}  category[known]\n        D                ...     ...              ...\n        G                ...     ...              ...\n        H                ...     ...              ...\n        {footer}')
    assert repr(ddf) == exp
    assert str(ddf) == exp
    exp_table = f'<table border="1" class="dataframe">\n  <thead>\n    <tr style="text-align: right;">\n      <th></th>\n      <th>A</th>\n      <th>B</th>\n      <th>C</th>\n    </tr>\n    <tr>\n      <th>npartitions=3</th>\n      <th></th>\n      <th></th>\n      <th></th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>A</th>\n      <td>int64</td>\n      <td>{string_dtype}</td>\n      <td>category[known]</td>\n    </tr>\n    <tr>\n      <th>D</th>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n    </tr>\n    <tr>\n      <th>G</th>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n    </tr>\n    <tr>\n      <th>H</th>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n    </tr>\n  </tbody>\n</table>'
    exp = f'<div><strong>Dask DataFrame Structure:</strong></div>\n{exp_table}\n<div>{footer}</div>'
    assert ddf.to_html() == exp
    exp = f'<div><strong>Dask DataFrame Structure:</strong></div>\n<div>\n{style}{exp_table}\n</div>\n<div>{footer}</div>'
    assert ddf._repr_html_() == exp