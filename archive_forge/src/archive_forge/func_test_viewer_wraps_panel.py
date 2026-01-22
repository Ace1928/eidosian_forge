import param
import pytest
from panel import config
from panel.interact import interactive
from panel.pane import Markdown, Str, panel
from panel.param import ParamMethod
from panel.viewable import Viewable, Viewer
from .util import jb_available
def test_viewer_wraps_panel():
    tv = ExampleViewer(value='hello')
    view = tv._create_view()
    assert isinstance(view, Markdown)
    assert view.object == 'hello'