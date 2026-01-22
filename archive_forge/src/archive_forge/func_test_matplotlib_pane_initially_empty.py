from bokeh.models import Div, Row as BkRow
import panel as pn
from panel.pane import Bokeh, Matplotlib, PaneBase
from panel.tests.util import mpl_available, mpl_figure
@mpl_available
def test_matplotlib_pane_initially_empty(document, comm):
    pane = pn.pane.Matplotlib()
    assert pane.object is None
    model = pane.get_root(document, comm=comm)
    assert model.text == '<img></img>'
    pane.object = mpl_figure()
    assert model.text.startswith('&lt;img src=&quot;data:image/png;base64,')
    assert pane._models[model.ref['id']][0] is model