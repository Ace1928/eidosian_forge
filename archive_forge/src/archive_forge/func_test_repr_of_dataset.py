from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.core import formatting_html as fh
from xarray.core.coordinates import Coordinates
def test_repr_of_dataset(dataset: xr.Dataset) -> None:
    formatted = fh.dataset_repr(dataset)
    assert formatted.count("class='xr-section-summary-in' type='checkbox'  checked>") == 3
    assert formatted.count("class='xr-section-summary-in' type='checkbox'  >") == 1
    assert '&lt;U4' in formatted or '&gt;U4' in formatted
    assert '&lt;IA&gt;' in formatted
    with xr.set_options(display_expand_coords=False, display_expand_data_vars=False, display_expand_attrs=False, display_expand_indexes=True):
        formatted = fh.dataset_repr(dataset)
        assert formatted.count("class='xr-section-summary-in' type='checkbox'  checked>") == 1
        assert '&lt;U4' in formatted or '&gt;U4' in formatted
        assert '&lt;IA&gt;' in formatted