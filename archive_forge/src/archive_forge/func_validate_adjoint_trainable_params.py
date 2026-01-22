import os
from typing import Generator, Callable, Union, Sequence, Optional
from copy import copy
import warnings
import pennylane as qml
from pennylane import Snapshot
from pennylane.operation import Tensor, StatePrepBase
from pennylane.measurements import (
from pennylane.typing import ResultBatch, Result
from pennylane import DeviceError
from pennylane import transform
from pennylane.wires import WireError
@transform
def validate_adjoint_trainable_params(tape: qml.tape.QuantumTape) -> (Sequence[qml.tape.QuantumTape], Callable):
    """Raises a warning if any of the observables is trainable, and raises an error if any
    trainable parameters belong to state-prep operations. Can be used in validating circuits
    for adjoint differentiation.
    """
    for op in tape.operations[:tape.num_preps]:
        if qml.operation.is_trainable(op):
            raise qml.QuantumFunctionError('Differentiating with respect to the input parameters of state-prep operations is not supported with the adjoint differentiation method.')
    for k in tape.trainable_params:
        mp_or_op = tape[tape._par_info[k]['op_idx']]
        if isinstance(mp_or_op, MeasurementProcess):
            warnings.warn(f'Differentiating with respect to the input parameters of {mp_or_op.obs.name} is not supported with the adjoint differentiation method. Gradients are computed only with regards to the trainable parameters of the circuit.\n\n Mark the parameters of the measured observables as non-trainable to silence this warning.', UserWarning)
    return ((tape,), null_postprocessing)