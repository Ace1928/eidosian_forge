from unittest import mock
import pytest
from cirq.circuits import TextDiagramDrawer
from cirq.circuits._block_diagram_drawer_test import _assert_same_diagram
from cirq.circuits._box_drawing_character_data import (
from cirq.circuits.text_diagram_drawer import (
import cirq.testing as ct
def test_drawer_stack():
    d = TextDiagramDrawer()
    d.write(0, 0, 'A')
    d.write(1, 0, 'B')
    d.write(1, 1, 'C')
    dd = TextDiagramDrawer()
    dd.write(0, 0, 'D')
    dd.write(0, 1, 'E')
    dd.write(1, 1, 'F')
    vstacked = TextDiagramDrawer.vstack((dd, d))
    expected = '\nD\n\nE F\n\nA B\n\n  C\n    '.strip()
    assert_has_rendering(vstacked, expected)
    hstacked = TextDiagramDrawer.hstack((d, dd))
    expected = '\nA B D\n\n  C E F\n    '.strip()
    assert_has_rendering(hstacked, expected)
    d.force_horizontal_padding_after(0, 0)
    with pytest.raises(ValueError):
        TextDiagramDrawer.vstack((dd, d))
    dd.force_horizontal_padding_after(0, 0)
    expected = '\nD\n\nEF\n\nAB\n\n C\n    '.strip()
    vstacked = TextDiagramDrawer.vstack((dd, d))
    assert_has_rendering(vstacked, expected)
    d.force_vertical_padding_after(0, 0)
    with pytest.raises(ValueError):
        print(d.vertical_padding)
        print(dd.vertical_padding)
        TextDiagramDrawer.hstack((d, dd))
    dd.force_vertical_padding_after(0, 0)
    expected = '\nAB D\n C EF\n    '.strip()
    hstacked = TextDiagramDrawer.hstack((d, dd))
    assert_has_rendering(hstacked, expected)
    d.force_horizontal_padding_after(0, 0)
    dd.force_horizontal_padding_after(0, 2)
    d.force_vertical_padding_after(0, 1)
    dd.force_vertical_padding_after(0, 3)
    with pytest.raises(ValueError):
        TextDiagramDrawer.vstack((d, dd))
    vstacked = TextDiagramDrawer.vstack((dd, d), padding_resolver=max)
    expected = '\nD\n\n\n\nE  F\n\nA  B\n\n   C\n    '.strip()
    assert_has_rendering(vstacked, expected)
    hstacked = TextDiagramDrawer.hstack((d, dd), padding_resolver=max)
    expected = '\nAB D\n\n\n\n C E  F\n    '.strip()
    assert_has_rendering(hstacked, expected)
    vstacked_min = TextDiagramDrawer.vstack((dd, d), padding_resolver=min)
    expected = '\nD\n\n\n\nEF\n\nAB\n\n C\n    '.strip()
    assert_has_rendering(vstacked_min, expected)
    hstacked_min = TextDiagramDrawer.hstack((d, dd), padding_resolver=min)
    expected = '\nAB D\n\n C E  F\n    '.strip()
    assert_has_rendering(hstacked_min, expected)