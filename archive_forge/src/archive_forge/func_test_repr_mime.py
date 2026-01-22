from math import pi
import pytest
from IPython import get_ipython
from traitlets.config import Config
from IPython.core.formatters import (
from IPython.utils.io import capture_output
def test_repr_mime():

    class HasReprMime(object):

        def _repr_mimebundle_(self, include=None, exclude=None):
            return {'application/json+test.v2': {'x': 'y'}, 'plain/text': '<HasReprMime>', 'image/png': 'i-overwrite'}

        def _repr_png_(self):
            return 'should-be-overwritten'

        def _repr_html_(self):
            return '<b>hi!</b>'
    f = get_ipython().display_formatter
    html_f = f.formatters['text/html']
    save_enabled = html_f.enabled
    html_f.enabled = True
    obj = HasReprMime()
    d, md = f.format(obj)
    html_f.enabled = save_enabled
    assert sorted(d) == ['application/json+test.v2', 'image/png', 'plain/text', 'text/html', 'text/plain']
    assert md == {}
    d, md = f.format(obj, include={'image/png'})
    assert list(d.keys()) == ['image/png'], 'Include should filter out even things from repr_mimebundle'
    assert d['image/png'] == 'i-overwrite', '_repr_mimebundle_ take precedence'