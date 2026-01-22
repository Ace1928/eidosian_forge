from typing import Union
from copy import copy
import pennylane as qml
import pennylane.math as qnp
from pennylane.operation import Operator, convert_to_opmath
from pennylane.ops.op_math.pow import Pow
from pennylane.ops.op_math.sum import Sum
from pennylane.queuing import QueuingManager
from .symbolicop import ScalarSymbolicOp
Reduce the depth of nested operators to the minimum.

        Returns:
            .Operator: simplified operator
        