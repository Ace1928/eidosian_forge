from __future__ import annotations
import copy
from numbers import Number
import numpy as np
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.operators.linear_op import LinearOp
from qiskit.quantum_info.operators.operator import Operator
from qiskit.quantum_info.operators.mixins import generate_apidocs
Pad another operator with identities.

        Args:
            current (BaseOperator): current operator.
            other (BaseOperator): other operator.
            qargs (None or list): qargs

        Returns:
            BaseOperator: the padded operator.
        