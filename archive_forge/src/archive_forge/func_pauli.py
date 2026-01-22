import cmath
import math
import numbers
from typing import (
import numpy as np
import sympy
import cirq
from cirq import value, protocols, linalg, qis
from cirq._doc import document
from cirq._import import LazyLoader
from cirq.ops import (
from cirq.type_workarounds import NotImplementedType
@property
def pauli(self) -> pauli_gates.Pauli:
    return cast(pauli_gates.Pauli, self.gate)