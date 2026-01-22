from typing import Callable, cast, Dict, Union
import numpy as np
from pyquil.parser import parse
from pyquil.quilbase import (
from cirq import Circuit, LineQubit
from cirq.ops import (
Convert a Quil program to a Cirq Circuit.

    Args:
        quil: The Quil program to convert.

    Returns:
        A Cirq Circuit generated from the Quil program.

    Raises:
        UnsupportedQuilInstruction: Cirq does not support the specified Quil instruction.
        UndefinedQuilGate: Cirq does not support the specified Quil gate.

    References:
        https://github.com/rigetti/pyquil
    