import numpy
from qiskit.exceptions import QiskitError
from qiskit.circuit.exceptions import CircuitError
from .parametervector import ParameterVectorElement
def sort_parameters(parameters):
    """Sort an iterable of :class:`.Parameter` instances into a canonical order, respecting the
    ordering relationships between elements of :class:`.ParameterVector`\\ s."""

    def key(parameter):
        if isinstance(parameter, ParameterVectorElement):
            return (parameter.vector.name, parameter.index)
        return (parameter.name,)
    return sorted(parameters, key=key)