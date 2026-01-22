from typing import List, Dict, Union
import warnings
import numpy as np
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.pulse.exceptions import UnassignedDurationError, QiskitError
def format_parameter_value(operand: ParameterExpression, decimal: int=10) -> Union[ParameterExpression, complex]:
    """Convert ParameterExpression into the most suitable data type.

    Args:
        operand: Operand value in arbitrary data type including ParameterExpression.
        decimal: Number of digit to round returned value.

    Returns:
        Value casted to non-parameter data type, when possible.
    """
    try:
        evaluated = complex(operand)
        evaluated = np.round(evaluated, decimals=decimal)
        if np.isreal(evaluated):
            evaluated = float(evaluated.real)
            if evaluated.is_integer():
                evaluated = int(evaluated)
        else:
            warnings.warn("Assignment of complex values to ParameterExpression in Qiskit Pulse objects is now pending deprecation. This will align the Pulse module with other modules where such assignment wasn't possible to begin with. The typical use case for complex parameters in the module was the SymbolicPulse library. As of Qiskit-Terra 0.23.0 all library pulses were converted from complex amplitude representation to real representation using two floats (amp,angle), as used in the ScalableSymbolicPulse class. This eliminated the need for complex parameters. Any use of complex parameters (and particularly custom-built pulses) should be converted in a similar fashion to avoid the use of complex parameters.", PendingDeprecationWarning)
        return evaluated
    except TypeError:
        pass
    return operand