from __future__ import annotations
import os.path
import pytest
from dask.utils import format_bytes
from dask.widgets import FILTERS, TEMPLATE_PATHS, get_environment, get_template
def test_widgets():
    template = get_template('example.html.j2')
    assert isinstance(template, jinja2.Template)
    rendered = template.render(foo='bar')
    assert 'Hello bar' in rendered