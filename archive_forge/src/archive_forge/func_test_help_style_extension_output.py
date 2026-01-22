import hvplot.pandas
import pytest
from holoviews.core import Store
from holoviews.element import Curve
def test_help_style_extension_output(reset_default_backend):
    docstring, signature = hvplot._get_doc_and_signature(cls=hvplot.hvPlot, kind='line', completions=False, docstring=False, generic=False, style=True, signature=None)
    assert docstring == '\nStyle options\n-------------\n\n' + '\n'.join(sorted(Store.registry['bokeh'][Curve].style_opts))
    hvplot.extension('matplotlib', 'plotly')
    docstring, signature = hvplot._get_doc_and_signature(cls=hvplot.hvPlot, kind='line', completions=False, docstring=False, generic=False, style=True, signature=None)
    assert docstring == '\nStyle options\n-------------\n\n' + '\n'.join(sorted(Store.registry['matplotlib'][Curve].style_opts))
    hvplot.output(backend='plotly')
    docstring, signature = hvplot._get_doc_and_signature(cls=hvplot.hvPlot, kind='line', completions=False, docstring=False, generic=False, style=True, signature=None)
    assert docstring == '\nStyle options\n-------------\n\n' + '\n'.join(sorted(Store.registry['plotly'][Curve].style_opts))