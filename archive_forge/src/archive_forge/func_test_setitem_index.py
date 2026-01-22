from unittest import TestCase
from unittest import mock
import pytest
import traitlets
import ipywidgets as widgets
from ipywidgets.widgets.widget_templates import LayoutProperties
def test_setitem_index(self):
    box = widgets.GridspecLayout(2, 3)
    button1 = widgets.Button()
    button2 = widgets.Button()
    button3 = widgets.Button()
    button4 = widgets.Button()
    box[0, 0] = button1
    button1_label = button1.layout.grid_area
    assert button1 in box.children
    assert box.layout.grid_template_areas == '"{} . ."\n". . ."'.format(button1_label)
    box[-1, -1] = button2
    button2_label = button2.layout.grid_area
    assert button1_label != button2_label
    assert button2 in box.children
    assert box.layout.grid_template_areas == '"{} . ."\n". . {}"'.format(button1_label, button2_label)
    box[1, 0] = button3
    button3_label = button3.layout.grid_area
    assert button1_label != button3_label
    assert button2_label != button3_label
    assert button3 in box.children
    assert box.layout.grid_template_areas == '"{b1} . ."\n"{b3} . {b2}"'.format(b1=button1_label, b2=button2_label, b3=button3_label)
    box[1, 0] = button4
    button4_label = button4.layout.grid_area
    assert button1_label != button4_label
    assert button2_label != button4_label
    assert button4 in box.children
    assert button3 not in box.children
    assert box.layout.grid_template_areas == '"{b1} . ."\n"{b4} . {b2}"'.format(b1=button1_label, b2=button2_label, b4=button4_label)