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
def mid_circuit_measurements(tape: qml.tape.QuantumTape, device) -> (Sequence[qml.tape.QuantumTape], Callable):
    """Provide the transform to handle mid-circuit measurements.

    If the tape or device uses finite-shot, use the native implementation (i.e. no transform),
    and use the ``qml.defer_measurements`` transform otherwise.
    """
    if tape.shots and tape.batch_size is None:
        return ((tape,), null_postprocessing)
    return qml.defer_measurements(tape, device=device)