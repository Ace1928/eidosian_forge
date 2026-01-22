from typing import Any, Dict, Iterable, List, Optional, Sequence, Union
from collections import defaultdict
import itertools
import random
import numpy as np
import sympy
from cirq import circuits, ops, linalg, protocols, qis
from cirq.testing import lin_alg_utils
def assert_same_circuits(actual: circuits.AbstractCircuit, expected: circuits.AbstractCircuit) -> None:
    """Asserts that two circuits are identical, with a descriptive error.

    Args:
        actual: A circuit computed by some code under test.
        expected: The circuit that should have been computed.
    """
    assert actual == expected, f'Actual circuit differs from expected circuit.\n\nDiagram of actual circuit:\n{actual}\n\nDiagram of expected circuit:\n{expected}\n\nIndex of first differing moment:\n{_first_differing_moment_index(actual, expected)}\n\nFull repr of actual circuit:\n{actual!r}\n\nFull repr of expected circuit:\n{expected!r}\n'