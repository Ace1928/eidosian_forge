from unittest import mock
import pytest
from cirq.circuits import TextDiagramDrawer
from cirq.circuits._block_diagram_drawer_test import _assert_same_diagram
from cirq.circuits._box_drawing_character_data import (
from cirq.circuits.text_diagram_drawer import (
import cirq.testing as ct
def assert_has_rendering(actual: TextDiagramDrawer, desired: str, **kwargs) -> None:
    """Determines if a given diagram has the desired rendering.

    Args:
        actual: The text diagram.
        desired: The desired rendering as a string.
        **kwargs: Keyword arguments to be passed to actual.render.
    """
    actual_diagram = actual.render(**kwargs)
    desired_diagram = desired
    assert actual_diagram == desired_diagram, f"Diagram's rendering differs from the desired rendering.\n\nActual rendering:\n{actual_diagram}\n\nDesired rendering:\n{desired_diagram}\n\nHighlighted differences:\n{ct.highlight_text_differences(actual_diagram, desired_diagram)}\n"