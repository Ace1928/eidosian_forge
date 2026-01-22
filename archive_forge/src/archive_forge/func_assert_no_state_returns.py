from functools import partial
import warnings
import pennylane as qml
from pennylane.transforms.tape_expand import expand_invalid_trainable
from pennylane.measurements import (
def assert_no_state_returns(measurements, transform_name):
    """Check whether a set of measurements contains a measurement process that returns the quantum
    state and raise an error if this is the case.

    Args:
        measurements (list[MeasurementProcess]): measurements to analyze
        transform_name (str): Name of the gradient transform that queries the measurements

    Currently, the measurement processes that are considered to return the state are
    ``~.measurements.StateMP``, ``~.measurements.VnEntropyMP``, and ``~.measurements.MutualInfoMP``.
    """
    if any((isinstance(m, (StateMP, VnEntropyMP, MutualInfoMP)) for m in measurements)):
        raise ValueError(f'Computing the gradient of circuits that return the state with the {transform_name} gradient transform is not supported, as it is a hardware-compatible method.')