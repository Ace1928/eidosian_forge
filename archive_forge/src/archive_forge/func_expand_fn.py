import contextlib
import pennylane as qml
from pennylane.operation import (
def expand_fn(tape, depth=depth, **kwargs):
    with qml.QueuingManager.stop_recording():
        if stop_at is None:
            tape = tape.expand(depth=depth)
        elif not all((stop_at(op) for op in tape.operations)):
            tape = tape.expand(depth=depth, stop_at=stop_at)
        else:
            return tape
        _update_trainable_params(tape)
    return tape