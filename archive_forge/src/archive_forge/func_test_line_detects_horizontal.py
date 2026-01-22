from unittest import mock
import pytest
from cirq.circuits import TextDiagramDrawer
from cirq.circuits._block_diagram_drawer_test import _assert_same_diagram
from cirq.circuits._box_drawing_character_data import (
from cirq.circuits.text_diagram_drawer import (
import cirq.testing as ct
def test_line_detects_horizontal():
    d = TextDiagramDrawer()
    with mock.patch.object(d, 'vertical_line') as vertical_line:
        d.grid_line(1, 2, 1, 5, True)
        vertical_line.assert_called_once_with(1, 2, 5, True, False)