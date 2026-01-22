from unittest import mock
import pytest
from cirq.circuits import TextDiagramDrawer
from cirq.circuits._block_diagram_drawer_test import _assert_same_diagram
from cirq.circuits._box_drawing_character_data import (
from cirq.circuits.text_diagram_drawer import (
import cirq.testing as ct
def test_drawer_copy():
    orig_entries = {(0, 0): _DiagramText('entry', '')}
    orig_vertical_lines = [_VerticalLine(1, 1, 3, True, False)]
    orig_horizontal_lines = [_HorizontalLine(0, 0, 3, False, False)]
    orig_vertical_padding = {0: 2}
    orig_horizontal_padding = {1: 3}
    kwargs = {'entries': orig_entries, 'vertical_lines': orig_vertical_lines, 'horizontal_lines': orig_horizontal_lines, 'vertical_padding': orig_vertical_padding, 'horizontal_padding': orig_horizontal_padding}
    orig_drawer = TextDiagramDrawer(**kwargs)
    same_drawer = TextDiagramDrawer(**kwargs)
    assert orig_drawer == same_drawer
    copy_drawer = orig_drawer.copy()
    assert orig_drawer == copy_drawer
    copy_drawer.write(0, 1, 'new_entry')
    assert copy_drawer != orig_drawer
    copy_drawer = orig_drawer.copy()
    copy_drawer.vertical_line(2, 1, 3)
    assert copy_drawer != orig_drawer
    copy_drawer = orig_drawer.copy()
    copy_drawer.horizontal_line(2, 1, 3)
    assert copy_drawer != orig_drawer
    copy_drawer = orig_drawer.copy()
    copy_drawer.force_horizontal_padding_after(1, 4)
    assert copy_drawer != orig_drawer
    copy_drawer = orig_drawer.copy()
    copy_drawer.force_vertical_padding_after(1, 4)
    assert copy_drawer != orig_drawer