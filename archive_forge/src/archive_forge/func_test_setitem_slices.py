from unittest import TestCase
from unittest import mock
import pytest
import traitlets
import ipywidgets as widgets
from ipywidgets.widgets.widget_templates import LayoutProperties
def test_setitem_slices(self):
    box = widgets.GridspecLayout(2, 3)
    button1 = widgets.Button()
    box[:2, 0] = button1
    assert len(box.children) == 1
    assert button1 in box.children
    button1_label = button1.layout.grid_area
    assert box.layout.grid_template_areas == '"{b1} . ."\n"{b1} . ."'.format(b1=button1_label)
    box = widgets.GridspecLayout(2, 3)
    button1 = widgets.Button()
    button2 = widgets.Button()
    box[:2, 1:] = button1
    assert len(box.children) == 1
    assert button1 in box.children
    button1_label = button1.layout.grid_area
    assert box.layout.grid_template_areas == '". {b1} {b1}"\n". {b1} {b1}"'.format(b1=button1_label)
    box[:2, 1:] = button2
    assert len(box.children) == 1
    assert button2 in box.children
    button2_label = button2.layout.grid_area
    assert box.layout.grid_template_areas == '". {b1} {b1}"\n". {b1} {b1}"'.format(b1=button2_label)