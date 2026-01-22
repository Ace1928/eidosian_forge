from __future__ import annotations
import cmath
import math
import io
import base64
import warnings
from typing import ClassVar, Optional, Type
import logging
import numpy as np
from qiskit.circuit import QuantumRegister, QuantumCircuit, Gate
from qiskit.circuit.library.standard_gates import CXGate, RXGate, RYGate, RZGate
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators import Operator
from qiskit.synthesis.two_qubit.weyl import transform_to_magic_basis
from qiskit.synthesis.one_qubit.one_qubit_decompose import (
from qiskit._accelerate import two_qubit_decompose
@staticmethod
def decomp0(target):
    """
        Decompose target :math:`\\sim U_d(x, y, z)` with :math:`0` uses of the basis gate.
        Result :math:`U_r` has trace:

        .. math::

            \\Big\\vert\\text{Tr}(U_r\\cdot U_\\text{target}^{\\dag})\\Big\\vert =
            4\\Big\\vert (\\cos(x)\\cos(y)\\cos(z)+ j \\sin(x)\\sin(y)\\sin(z)\\Big\\vert

        which is optimal for all targets and bases
        """
    U0l = target.K1l.dot(target.K2l)
    U0r = target.K1r.dot(target.K2r)
    return (U0r, U0l)