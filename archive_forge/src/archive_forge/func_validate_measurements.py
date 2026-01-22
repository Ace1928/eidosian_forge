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
def validate_measurements(tape: qml.tape.QuantumTape, analytic_measurements=None, sample_measurements=None, name='device') -> (Sequence[qml.tape.QuantumTape], Callable):
    """Validates the supported state and sample based measurement processes.

    Args:
        tape (QuantumTape, .QNode, Callable): a quantum circuit.
        analytic_measurements (Callable[[MeasurementProcess], bool]): a function from a measurement process
            to whether or not it is accepted in analytic simulations.
        sample_measurements (Callable[[MeasurementProcess], bool]): a function from a measurement process
            to whether or not it accepted for finite shot siutations
        name (str): the name to use in error messages.

    Returns:
        qnode (pennylane.QNode) or quantum function (callable) or tuple[List[.QuantumTape], function]:

        The unaltered input circuit. The output type is explained in :func:`qml.transform <pennylane.transform>`.

    Raises:
        DeviceError: if a measurement process is not supported.

    >>> def analytic_measurements(m):
    ...     return isinstance(m, qml.measurements.StateMP)
    >>> def shots_measurements(m):
    ...     return isinstance(m, qml.measurements.CountsMP)
    >>> tape = qml.tape.QuantumScript([], [qml.expval(qml.Z(0))])
    >>> validate_measurements(tape, analytic_measurements, shots_measurements)
    DeviceError: Measurement expval(Z(0)) not accepted for analytic simulation on device.
    >>> tape = qml.tape.QuantumScript([], [qml.sample()], shots=10)
    >>> validate_measurements(tape, analytic_measurements, shots_measurements)
    DeviceError: Measurement sample(wires=[]) not accepted with finite shots on device

    """
    if analytic_measurements is None:

        def analytic_measurements(m):
            return isinstance(m, StateMeasurement)
    if sample_measurements is None:

        def sample_measurements(m):
            return isinstance(m, SampleMeasurement)
    if tape.shots:
        for m in tape.measurements:
            if not sample_measurements(m):
                raise DeviceError(f'Measurement {m} not accepted with finite shots on {name}')
    else:
        for m in tape.measurements:
            if not analytic_measurements(m):
                raise DeviceError(f'Measurement {m} not accepted for analytic simulation on {name}.')
    return ((tape,), null_postprocessing)