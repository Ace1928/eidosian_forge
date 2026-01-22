from typing import List, Dict, Union
import warnings
import numpy as np
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.pulse.exceptions import UnassignedDurationError, QiskitError
def instruction_duration_validation(duration: int):
    """Validate instruction duration.

    Args:
        duration: Instruction duration value to validate.

    Raises:
        UnassignedDurationError: When duration is unassigned.
        QiskitError: When invalid duration is assigned.
    """
    if isinstance(duration, ParameterExpression):
        raise UnassignedDurationError('Instruction duration {} is not assigned. Please bind all durations to an integer value before playing in the Schedule, or use ScheduleBlock to align instructions with unassigned duration.'.format(repr(duration)))
    if not isinstance(duration, (int, np.integer)) or duration < 0:
        raise QiskitError(f'Instruction duration must be a non-negative integer, got {duration} instead.')