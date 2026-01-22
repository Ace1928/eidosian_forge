import abc
import dataclasses
from typing import Iterable, List, TYPE_CHECKING
from cirq.ops import raw_types
Free pre-allocated clean or dirty qubits managed by this qubit manager.