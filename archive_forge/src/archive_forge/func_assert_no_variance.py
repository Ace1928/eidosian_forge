from functools import partial
import warnings
import pennylane as qml
from pennylane.transforms.tape_expand import expand_invalid_trainable
from pennylane.measurements import (
def assert_no_variance(measurements, transform_name):
    """Check whether a set of measurements contains a variance measurement
    raise an error if this is the case.

    Args:
        measurements (list[MeasurementProcess]): measurements to analyze
        transform_name (str): Name of the gradient transform that queries the measurements
    """
    if any((isinstance(m, VarianceMP) for m in measurements)):
        raise ValueError(f'Computing the gradient of variances with the {transform_name} gradient transform is not supported.')