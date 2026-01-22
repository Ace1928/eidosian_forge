import asyncio
import datetime as dt
import numpy as np
import pandas as pd
import pytest
from bokeh.models.widgets.tables import (
from packaging.version import Version
from panel.depends import bind
from panel.io.state import set_curdoc
from panel.models.tabulator import CellClickEvent, TableEditEvent
from panel.tests.util import mpl_available, serve_and_request, wait_until
from panel.util import BOKEH_JS_NAT
from panel.widgets import Button, TextInput
from panel.widgets.tables import DataFrame, Tabulator
def test_hierarchical_index(document, comm):
    df = pd.DataFrame([('Germany', 2020, 9, 2.4, 'A'), ('Germany', 2021, 3, 7.3, 'C'), ('Germany', 2022, 6, 3.1, 'B'), ('UK', 2020, 5, 8.0, 'A'), ('UK', 2021, 1, 3.9, 'B'), ('UK', 2022, 9, 2.2, 'A')], columns=['Country', 'Year', 'Int', 'Float', 'Str']).set_index(['Country', 'Year'])
    table = DataFrame(value=df, hierarchical=True, aggregators={'Year': {'Int': 'sum', 'Float': 'mean'}})
    model = table.get_root(document, comm)
    assert isinstance(model, DataCube)
    assert len(model.grouping) == 1
    grouping = model.grouping[0]
    assert len(grouping.aggregators) == 2
    agg1, agg2 = grouping.aggregators
    assert agg1.field_ == 'Int'
    assert isinstance(agg1, SumAggregator)
    assert agg2.field_ == 'Float'
    assert isinstance(agg2, AvgAggregator)
    table.aggregators = {'Year': 'min'}
    agg1, agg2 = grouping.aggregators
    assert agg1.field_ == 'Int'
    assert isinstance(agg1, MinAggregator)
    assert agg2.field_ == 'Float'
    assert isinstance(agg2, MinAggregator)