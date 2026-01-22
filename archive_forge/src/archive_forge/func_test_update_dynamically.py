from unittest import TestCase
from unittest import mock
import pytest
import traitlets
import ipywidgets as widgets
from ipywidgets.widgets.widget_templates import LayoutProperties
@mock.patch('ipywidgets.Layout.send_state')
def test_update_dynamically(self, send_state):
    """test whether it's possible to add widget outside __init__"""
    button1 = widgets.Button()
    button2 = widgets.Button()
    button3 = widgets.Button()
    button4 = widgets.Button()
    box = widgets.TwoByTwoLayout(top_left=button1, top_right=button3, bottom_left=None, bottom_right=button4)
    from ipykernel.kernelbase import Kernel
    state = box.get_state()
    assert len(state['children']) == 3
    assert box.layout.grid_template_areas == '"top-left top-right"\n' + '"top-left bottom-right"'
    box.layout.comm.kernel = mock.MagicMock(spec=Kernel)
    send_state.reset_mock()
    box.bottom_left = button2
    state = box.get_state()
    assert len(state['children']) == 4
    assert box.layout.grid_template_areas == '"top-left top-right"\n' + '"bottom-left bottom-right"'
    send_state.assert_called_with(key='grid_template_areas')
    box = widgets.TwoByTwoLayout(top_left=button1, top_right=button3, bottom_left=None, bottom_right=button4)
    assert box.layout.grid_template_areas == '"top-left top-right"\n' + '"top-left bottom-right"'
    box.layout.comm.kernel = mock.MagicMock(spec=Kernel)
    send_state.reset_mock()
    box.merge = False
    assert box.layout.grid_template_areas == '"top-left top-right"\n' + '"bottom-left bottom-right"'
    send_state.assert_called_with(key='grid_template_areas')