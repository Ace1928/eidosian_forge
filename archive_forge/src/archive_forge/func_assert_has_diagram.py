from typing import Any, Dict, Iterable, List, Optional, Sequence, Union
from collections import defaultdict
import itertools
import random
import numpy as np
import sympy
from cirq import circuits, ops, linalg, protocols, qis
from cirq.testing import lin_alg_utils
def assert_has_diagram(actual: Union[circuits.AbstractCircuit, circuits.Moment], desired: str, **kwargs) -> None:
    """Determines if a given circuit has the desired text diagram.

    Args:
        actual: The circuit that was actually computed by some process.
        desired: The desired text diagram as a string. Newlines at the
            beginning and whitespace at the end are ignored.
        **kwargs: Keyword arguments to be passed to actual.to_text_diagram().
    """
    __tracebackhide__ = True
    actual_diagram = actual.to_text_diagram(**kwargs).lstrip('\n').rstrip()
    desired_diagram = desired.lstrip('\n').rstrip()
    assert actual_diagram == desired_diagram, f"Circuit's text diagram differs from the desired diagram.\n\nDiagram of actual circuit:\n{actual_diagram}\n\nDesired text diagram:\n{desired_diagram}\n\nHighlighted differences:\n{highlight_text_differences(actual_diagram, desired_diagram)}\n"