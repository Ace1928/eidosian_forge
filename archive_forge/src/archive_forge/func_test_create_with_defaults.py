from unittest import TestCase
from unittest import mock
import pytest
import traitlets
import ipywidgets as widgets
from ipywidgets.widgets.widget_templates import LayoutProperties
def test_create_with_defaults(self):
    """test creating with default values"""
    footer = widgets.Button()
    header = widgets.Button()
    center = widgets.Button()
    left_sidebar = widgets.Button()
    right_sidebar = widgets.Button()
    box = widgets.AppLayout(footer=footer, header=header, center=center, left_sidebar=left_sidebar, right_sidebar=right_sidebar)
    assert box.layout.grid_template_areas == '"header header header"\n' + '"left-sidebar center right-sidebar"\n' + '"footer footer footer"'
    assert box.footer.layout.grid_area == 'footer'
    assert box.header.layout.grid_area == 'header'
    assert box.center.layout.grid_area == 'center'
    assert box.left_sidebar.layout.grid_area == 'left-sidebar'
    assert box.right_sidebar.layout.grid_area == 'right-sidebar'
    assert len(box.get_state()['children']) == 5
    box = widgets.AppLayout()
    assert box.layout.grid_template_areas is None
    assert box.layout.grid_template_columns is None
    assert box.layout.grid_template_rows is None
    assert len(box.get_state()['children']) == 0