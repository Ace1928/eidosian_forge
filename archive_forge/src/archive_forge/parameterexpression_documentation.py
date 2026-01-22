from __future__ import annotations
from typing import Callable, Union
import numbers
import operator
import numpy
import symengine
from qiskit.circuit.exceptions import CircuitError
Return symbolic expression as a raw Sympy or Symengine object.

        Symengine is used preferentially; if both are available, the result will always be a
        ``symengine`` object.  Symengine is a separate library but has integration with Sympy.

        .. note::

            This is for interoperability only.  Qiskit will not accept or work with raw Sympy or
            Symegine expressions in its parameters, because they do not contain the tracking
            information used in circuit-parameter binding and assignment.
        