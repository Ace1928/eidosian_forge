from unittest import mock
import pytest
from cirq.circuits import TextDiagramDrawer
from cirq.circuits._block_diagram_drawer_test import _assert_same_diagram
from cirq.circuits._box_drawing_character_data import (
from cirq.circuits.text_diagram_drawer import (
import cirq.testing as ct
def test_line_fails_when_not_aligned():
    d = TextDiagramDrawer()
    with pytest.raises(ValueError):
        d.grid_line(1, 2, 3, 4)