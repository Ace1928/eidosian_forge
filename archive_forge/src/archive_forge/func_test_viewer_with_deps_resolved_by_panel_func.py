import param
import pytest
from panel import config
from panel.interact import interactive
from panel.pane import Markdown, Str, panel
from panel.param import ParamMethod
from panel.viewable import Viewable, Viewer
from .util import jb_available
def test_viewer_with_deps_resolved_by_panel_func(document, comm):
    tv = ExampleViewerWithDeps(value='hello')
    view = panel(tv)
    view.get_root(document, comm)
    assert isinstance(view, ParamMethod)
    assert view._pane.object == 'hello'