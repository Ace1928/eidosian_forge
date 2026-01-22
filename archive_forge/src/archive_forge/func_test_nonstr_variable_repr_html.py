from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.core import formatting_html as fh
from xarray.core.coordinates import Coordinates
def test_nonstr_variable_repr_html() -> None:
    v = xr.Variable(['time', 10], [[1, 2, 3], [4, 5, 6]], {22: 'bar'})
    assert hasattr(v, '_repr_html_')
    with xr.set_options(display_style='html'):
        html = v._repr_html_().strip()
    assert '<dt><span>22 :</span></dt><dd>bar</dd>' in html
    assert '<li><span>10</span>: 3</li></ul>' in html