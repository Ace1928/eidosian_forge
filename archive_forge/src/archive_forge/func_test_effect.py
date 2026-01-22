import pytest
import traitlets
from bokeh.core.has_props import _default_resolver
from bokeh.model import Model
from panel.layout import Row
from panel.pane.ipywidget import Reacton
from panel.tests.util import serve_component, wait_until
def test_effect():
    runs.append(button)

    def cleanup():
        cleanups.append(button)
    return cleanup