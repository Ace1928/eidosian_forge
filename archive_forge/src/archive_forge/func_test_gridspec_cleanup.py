import pytest
from bokeh.models import Div
from panel.depends import depends
from panel.layout import GridBox, GridSpec, Spacer
from panel.widgets import IntSlider
def test_gridspec_cleanup(document, comm):
    spacer = Spacer()
    gspec = GridSpec()
    gspec[0, 0] = spacer
    model = gspec.get_root(document, comm)
    ref = model.ref['id']
    assert ref in gspec._models
    assert ref in spacer._models
    gspec._cleanup(model)
    assert ref not in gspec._models
    assert ref not in spacer._models