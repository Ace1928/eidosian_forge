from copy import deepcopy
import functools
import inspect
import os
import warnings
import pennylane as qml
from pennylane.tape import make_qscript
@functools.wraps(fn)
def internal_wrapper(*args, **kwargs):
    tape = make_qscript(fn)(*args, **kwargs)
    tape = tape_transform(tape, *transform_args, **transform_kwargs)
    num_measurements = len(tape.measurements)
    if num_measurements == 0:
        return None
    return tape.measurements[0] if num_measurements == 1 else tape.measurements