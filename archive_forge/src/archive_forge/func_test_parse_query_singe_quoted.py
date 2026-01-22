import sys
from collections import OrderedDict
import param
from bokeh.models import Div
from panel.depends import bind
from panel.io.notebook import render_mimebundle
from panel.pane import PaneBase
from panel.tests.util import mpl_available
from panel.util import (
def test_parse_query_singe_quoted():
    query = '?str=abc&json=%5B%27def%27%5D'
    expected_results = {'str': 'abc', 'json': ['def']}
    results = parse_query(query)
    assert expected_results == results