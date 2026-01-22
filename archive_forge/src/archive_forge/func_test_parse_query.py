import sys
from collections import OrderedDict
import param
from bokeh.models import Div
from panel.depends import bind
from panel.io.notebook import render_mimebundle
from panel.pane import PaneBase
from panel.tests.util import mpl_available
from panel.util import (
def test_parse_query():
    query = '?bool=true&int=2&float=3.0&json=["a"%2C+"b"]'
    expected_results = {'bool': True, 'int': 2, 'float': 3.0, 'json': ['a', 'b']}
    results = parse_query(query)
    assert expected_results == results