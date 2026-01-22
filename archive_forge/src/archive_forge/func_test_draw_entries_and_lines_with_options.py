from unittest import mock
import pytest
from cirq.circuits import TextDiagramDrawer
from cirq.circuits._block_diagram_drawer_test import _assert_same_diagram
from cirq.circuits._box_drawing_character_data import (
from cirq.circuits.text_diagram_drawer import (
import cirq.testing as ct
def test_draw_entries_and_lines_with_options():
    d = TextDiagramDrawer()
    d.write(0, 0, '!')
    d.write(6, 2, 'span')
    d.horizontal_line(y=3, x1=2, x2=8)
    d.vertical_line(x=7, y1=1, y2=4)
    _assert_same_diagram(d.render().strip(), '\n!\n\n                 ╷\n                 │\n            span │\n                 │\n    ╶────────────┼─\n                 │\n    '.strip())
    _assert_same_diagram(d.render(use_unicode_characters=False).strip(), '\n!\n\n\n                 |\n            span |\n                 |\n     ------------+-\n                 |\n    '.strip())
    _assert_same_diagram(d.render(crossing_char='@').strip(), '\n!\n\n                 ╷\n                 │\n            span │\n                 │\n    ╶────────────@─\n                 │\n    '.strip())
    _assert_same_diagram(d.render(horizontal_spacing=0).strip(), '\n!\n\n          ╷\n          │\n      span│\n          │\n  ╶───────┼\n          │\n    '.strip())
    _assert_same_diagram(d.render(vertical_spacing=0).strip(), '\n!\n                 ╷\n            span │\n    ╶────────────┼─\n    '.strip())