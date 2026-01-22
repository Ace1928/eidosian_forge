from packaging.version import Version
import holoviews as hv
import hvplot.pandas  # noqa
import hvplot.xarray  # noqa
import matplotlib
import numpy as np
import pandas as pd
import panel as pn
import pytest
import xarray as xr
from holoviews.util.transform import dim
from hvplot import bind
from hvplot.interactive import Interactive
from hvplot.tests.util import makeDataFrame, makeMixedDataFrame
from hvplot.xarray import XArrayInteractive
from hvplot.util import bokeh3, param2
def test_interactive_with_bound_function_calls():
    df = pd.DataFrame({'species': [1, 1, 1, 2, 2, 2], 'sex': 3 * ['MALE', 'FEMALE']})
    w_species = pn.widgets.Select(name='Species', options=[1, 2])
    w_sex = pn.widgets.MultiSelect(name='Sex', value=['MALE'], options=['MALE', 'FEMALE'])

    def load_data(species, watch=True):
        if watch:
            load_data.COUNT += 1
        return df.loc[df['species'] == species]
    load_data.COUNT = 0
    dfi = bind(load_data, w_species).interactive()
    dfi = dfi.loc[dfi['sex'].isin(w_sex)]
    out = dfi.output()
    assert isinstance(out, pn.param.ParamFunction)
    assert isinstance(out._pane, pn.pane.DataFrame)
    pd.testing.assert_frame_equal(out._pane.object, load_data(w_species.value, watch=False).loc[df['sex'].isin(w_sex.value)])
    dfi.loc[dfi['sex'].isin(w_sex)]
    assert load_data.COUNT == 1
    w_species.value = 2
    pd.testing.assert_frame_equal(out._pane.object, load_data(w_species.value, watch=False).loc[df['sex'].isin(w_sex.value)])
    assert load_data.COUNT == 2
    dfi = dfi.head(1)
    assert load_data.COUNT == 2
    out = dfi.output()
    pd.testing.assert_frame_equal(out._pane.object, load_data(w_species.value, watch=False).loc[df['sex'].isin(w_sex.value)].head(1))
    w_species.value = 1
    pd.testing.assert_frame_equal(out._pane.object, load_data(w_species.value, watch=False).loc[df['sex'].isin(w_sex.value)].head(1))
    assert load_data.COUNT == 3