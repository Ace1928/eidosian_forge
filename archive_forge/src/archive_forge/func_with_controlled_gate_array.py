import numpy
from qiskit.exceptions import QiskitError
from qiskit.circuit.exceptions import CircuitError
from .parametervector import ParameterVectorElement
def with_controlled_gate_array(base_array, num_ctrl_qubits, cached_states=None):
    """Class decorator that adds an ``__array__`` method to a :class:`.ControlledGate` instance that
    returns singleton nonwritable views onto a relevant precomputed complex matrix for the given
    control state.

    If ``cached_states`` is not given, then all possible control states are precomputed.  If it is
    given, it should be an iterable of integers, and only these control states will be cached."""
    base = numpy.asarray(base_array, dtype=numpy.complex128)

    def matrix_for_control_state(state):
        out = numpy.asarray(_compute_control_matrix(base, num_ctrl_qubits, state), dtype=numpy.complex128)
        out.setflags(write=False)
        return out
    if cached_states is None:
        nonwritables = [matrix_for_control_state(state) for state in range(2 ** num_ctrl_qubits)]

        def __array__(self, dtype=None):
            return numpy.asarray(nonwritables[self.ctrl_state], dtype=dtype)
    else:
        nonwritables = {state: matrix_for_control_state(state) for state in cached_states}

        def __array__(self, dtype=None):
            if (out := nonwritables.get(self.ctrl_state)) is not None:
                return numpy.asarray(out, dtype=dtype)
            return numpy.asarray(_compute_control_matrix(base, num_ctrl_qubits, self.ctrl_state), dtype=dtype)

    def decorator(cls):
        if hasattr(cls, '__array__'):
            raise RuntimeError("Refusing to decorate a class that already has '__array__' defined.")
        cls.__array__ = __array__
        return cls
    return decorator