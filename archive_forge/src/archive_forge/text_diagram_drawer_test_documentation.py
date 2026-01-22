from unittest import mock
import pytest
from cirq.circuits import TextDiagramDrawer
from cirq.circuits._block_diagram_drawer_test import _assert_same_diagram
from cirq.circuits._box_drawing_character_data import (
from cirq.circuits.text_diagram_drawer import (
import cirq.testing as ct
Determines if a given diagram has the desired rendering.

    Args:
        actual: The text diagram.
        desired: The desired rendering as a string.
        **kwargs: Keyword arguments to be passed to actual.render.
    