import pytest
from bokeh.models import Div
from panel.depends import depends
from panel.layout import GridBox, GridSpec, Spacer
from panel.widgets import IntSlider
def test_gridspec_fixed_with_replacement_pane(document, comm):
    slider = IntSlider(start=0, end=2)

    @depends(slider)
    def div(value):
        return Div(text=str(value))
    gspec = GridSpec(width=600, height=600)
    gspec[0, 0:2] = Div()
    gspec[1, 2] = div
    model = gspec.get_root(document, comm=comm)
    (div1, _, _, _, _), (row, _, _, _, _) = model.children
    div2 = row.children[0]
    assert div1.width == 400
    assert div1.height == 300
    assert div2.width == 200
    assert div2.height == 300
    slider.value = 1
    assert row.children[0] is not div2
    assert row.children[0].width == 200
    assert row.children[0].height == 300