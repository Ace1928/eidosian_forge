import warnings
from typing import Sequence, Tuple, Union
import pennylane as qml
from pennylane.operation import Operator
from pennylane.wires import Wires
from .measurements import SampleMeasurement, StateMeasurement, Variance
from .mid_measure import MeasurementValue
Measurement process that computes the variance of the supplied observable.

    Please refer to :func:`var` for detailed documentation.

    Args:
        obs (Union[.Operator, .MeasurementValue]): The observable that is to be measured
            as part of the measurement process. Not all measurement processes require observables
            (for example ``Probability``); this argument is optional.
        wires (.Wires): The wires the measurement process applies to.
            This can only be specified if an observable was not provided.
        eigvals (array): A flat array representing the eigenvalues of the measurement.
            This can only be specified if an observable was not provided.
        id (str): custom label given to a measurement instance, can be useful for some applications
            where the instance has to be identified
    