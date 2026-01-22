from __future__ import annotations
import copy
import os
import re
import sys
from functools import partial
from operator import add, neg
import pytest
from dask.dot import _to_cytoscape_json, cytoscape_graph
from dask import delayed
from dask.utils import ensure_not_exists
@pytest.mark.parametrize('format,typ', [pytest.param('png', Image, marks=ipython_not_installed_mark), pytest.param('jpeg', Image, marks=pytest.mark.xfail(reason='jpeg not always supported in dot', strict=False)), ('dot', type(None)), ('pdf', type(None)), pytest.param('svg', SVG, marks=ipython_not_installed_mark)])
@pytest.mark.xfail(sys.platform == 'win32', reason='graphviz/pango on conda-forge currently broken for windows', strict=False)
def test_dot_graph_no_filename(tmpdir, format, typ):
    before = tmpdir.listdir()
    result = dot_graph(dsk, filename=None, format=format)
    after = tmpdir.listdir()
    assert before == after
    assert isinstance(result, typ)