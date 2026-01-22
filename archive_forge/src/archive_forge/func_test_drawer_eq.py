from unittest import mock
import pytest
from cirq.circuits import TextDiagramDrawer
from cirq.circuits._block_diagram_drawer_test import _assert_same_diagram
from cirq.circuits._box_drawing_character_data import (
from cirq.circuits.text_diagram_drawer import (
import cirq.testing as ct
def test_drawer_eq():
    assert TextDiagramDrawer().__eq__(23) == NotImplemented
    eq = ct.EqualsTester()
    d = TextDiagramDrawer()
    d.write(0, 0, 'A')
    d.write(1, 0, 'B')
    d.write(1, 1, 'C')
    alt_d = TextDiagramDrawer()
    alt_d.write(0, 0, 'A')
    alt_d.write(1, 0, 'B')
    alt_d.write(1, 1, 'C')
    eq.add_equality_group(d, alt_d)
    dd = TextDiagramDrawer()
    dd.write(0, 0, 'D')
    dd.write(0, 1, 'E')
    dd.write(1, 1, 'F')
    eq.add_equality_group(dd)