from functools import wraps
import inspect
from typing import Union, Callable, Tuple
import pennylane as qml
from .qnode import QNode, _make_execution_config, _get_device_shots
def batch_constructor(*args, **kwargs) -> Tuple[Tuple['qml.tape.QuantumTape', Callable]]:
    """Create a batch of tapes and a post processing function."""
    if 'shots' in inspect.signature(qnode.func).parameters:
        shots = _get_device_shots(qnode.device)
    else:
        shots = kwargs.pop('shots', _get_device_shots(qnode.device))
    initial_tape = qml.tape.make_qscript(qnode.func, shots=shots)(*args, **kwargs)
    return program((initial_tape,))