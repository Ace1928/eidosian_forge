from typing import Sequence, Union
import copy
from functools import singledispatch
import pennylane as qml
from pennylane.typing import TensorLike
from pennylane.operation import Operator, Tensor
from ..identity import Identity
from ..qubit import Projector
from ..op_math import CompositeOp, SymbolicOp, ScalarSymbolicOp, Adjoint, Pow, SProd
@bind_new_parameters.register
def bind_new_parameters_composite_op(op: CompositeOp, params: Sequence[TensorLike]):
    new_operands = []
    for operand in op.operands:
        op_num_params = operand.num_params
        sub_params = params[:op_num_params]
        params = params[op_num_params:]
        new_operands.append(bind_new_parameters(operand, sub_params))
    return op.__class__(*new_operands)