from __future__ import annotations
import os.path
import pytest
from dask.utils import format_bytes
from dask.widgets import FILTERS, TEMPLATE_PATHS, get_environment, get_template
def test_unknown_template():
    with pytest.raises(jinja2.TemplateNotFound) as e:
        get_template('does_not_exist.html.j2')
        assert os.path.dirname(os.path.abspath(__file__)) in str(e)