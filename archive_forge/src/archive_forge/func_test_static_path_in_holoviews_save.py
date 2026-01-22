import re
from io import StringIO
from pathlib import Path
import numpy as np
from bokeh.resources import Resources
from panel.config import config
from panel.io.resources import CDN_DIST
from panel.models.vega import VegaPlot
from panel.pane import Alert, Vega
from panel.tests.util import hv_available
@hv_available
def test_static_path_in_holoviews_save(tmpdir):
    import holoviews as hv
    hv.Store.set_current_backend('bokeh')
    plot = hv.Curve(np.random.seed(42))
    res = Resources(mode='server', root_url='/')
    out_file = Path(tmpdir) / 'plot.html'
    hv.save(plot, out_file, resources=res)
    content = out_file.read_text()
    assert 'src="/static/js/bokeh' in content and 'src="static/js/bokeh' not in content