from unittest import TestCase
from unittest import mock
import pytest
import traitlets
import ipywidgets as widgets
from ipywidgets.widgets.widget_templates import LayoutProperties
def test_set_pane_widths_heights(self):
    footer = widgets.Button()
    header = widgets.Button()
    center = widgets.Button()
    left_sidebar = widgets.Button()
    right_sidebar = widgets.Button()
    box = widgets.AppLayout(header=header, footer=footer, left_sidebar=left_sidebar, right_sidebar=left_sidebar, center=center)
    with pytest.raises(traitlets.TraitError):
        box.pane_widths = ['1fx', '1fx', '1fx', '1fx']
    with pytest.raises(traitlets.TraitError):
        box.pane_widths = ['1fx', '1fx']
    with pytest.raises(traitlets.TraitError):
        box.pane_heights = ['1fx', '1fx', '1fx', '1fx']
    with pytest.raises(traitlets.TraitError):
        box.pane_heights = ['1fx', '1fx']
    assert box.layout.grid_template_rows == '1fr 3fr 1fr'
    assert box.layout.grid_template_columns == '1fr 2fr 1fr'
    box.pane_heights = ['3fr', '100px', 20]
    assert box.layout.grid_template_rows == '3fr 100px 20fr'
    assert box.layout.grid_template_columns == '1fr 2fr 1fr'
    box.pane_widths = [3, 3, 1]
    assert box.layout.grid_template_rows == '3fr 100px 20fr'
    assert box.layout.grid_template_columns == '3fr 3fr 1fr'