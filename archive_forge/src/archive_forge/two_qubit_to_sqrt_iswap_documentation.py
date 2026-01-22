from typing import Optional, Sequence, Tuple, TYPE_CHECKING
import numpy as np
import sympy
from cirq import circuits, ops, linalg, protocols
from cirq.transformers.analytical_decompositions import single_qubit_decompositions
from cirq.transformers.merge_single_qubit_gates import merge_single_qubit_gates_to_phxz
Appends the decomposed single-qubit operations for matrix0 and
        matrix1.

        The cleanup logic, specific to sqrt-iSWAP, commutes the final Z**a gate
        and any whole X or Y gate on q1 through the following sqrt-iSWAP.

        Commutation rules:
        - Z(q0)**a, Z(q1)**a together commute with sqrt-iSWAP for all a
        - X(q0), X(q0) together commute with sqrt-iSWAP
        - Y(q0), Y(q0) together commute with sqrt-iSWAP
        