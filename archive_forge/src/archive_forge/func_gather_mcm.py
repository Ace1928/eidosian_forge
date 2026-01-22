from collections import Counter
from typing import Optional, Sequence
import warnings
from numpy.random import default_rng
import numpy as np
import pennylane as qml
from pennylane.measurements import (
from pennylane.typing import Result
from .initialize_state import create_initial_state
from .apply_operation import apply_operation
from .measure import measure
from .sampling import measure_with_samples
def gather_mcm(measurement, samples):
    """Combines, gathers and normalizes several measurements with non-trivial measurement values.

    Args:
        measurement (MeasurementProcess): measurement
        samples (List[dict]): Mid-circuit measurement samples

    Returns:
        TensorLike: The combined measurement outcome
    """
    mv = measurement.mv
    if isinstance(measurement, (CountsMP, ProbabilityMP, SampleMP)) and isinstance(mv, Sequence):
        wires = qml.wires.Wires(range(len(mv)))
        mcm_samples = list((np.array([m.concretize(dct) for dct in samples]).reshape((-1, 1)) for m in mv))
        mcm_samples = np.concatenate(mcm_samples, axis=1)
        meas_tmp = measurement.__class__(wires=wires)
        return meas_tmp.process_samples(mcm_samples, wire_order=wires)
    mcm_samples = np.array([mv.concretize(dct) for dct in samples]).reshape((-1, 1))
    use_as_is = len(mv.measurements) == 1
    if use_as_is:
        wires, meas_tmp = (mv.wires, measurement)
    else:
        if isinstance(measurement, (ExpectationMP, VarianceMP)):
            mcm_samples = mcm_samples.ravel()
        wires = qml.wires.Wires(0)
        meas_tmp = measurement.__class__(wires=wires)
    new_measurement = meas_tmp.process_samples(mcm_samples, wire_order=wires)
    if isinstance(measurement, CountsMP) and (not use_as_is):
        new_measurement = dict(sorted(((int(x, 2), y) for x, y in new_measurement.items())))
    return new_measurement