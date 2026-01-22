from __future__ import annotations
from typing import Optional, List, Tuple, Union, Iterable
import qiskit.circuit
from qiskit.circuit import Barrier, Delay
from qiskit.circuit import Instruction, ParameterExpression
from qiskit.circuit.duration import duration_in_dt
from qiskit.providers import Backend
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.utils.units import apply_prefix
Get the set of all units used in this instruction durations.

        Returns:
            Set of units used in this instruction durations.
        