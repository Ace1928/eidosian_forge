import functools
import os
import copy
import warnings
import types
from typing import Sequence
import pennylane as qml
from pennylane.typing import ResultBatch
def qfunc_transformed(*args, **kwargs):
    with qml.queuing.AnnotatedQueue() as q:
        qfunc_output = qfunc(*args, **kwargs)
    tape = qml.tape.QuantumScript.from_queue(q)
    with qml.QueuingManager.stop_recording():
        transformed_tapes, processing_fn = self._transform(tape, *targs, **tkwargs)
    if len(transformed_tapes) != 1:
        raise TransformError('Impossible to dispatch your transform on quantum function, because more than one tape is returned')
    transformed_tape = transformed_tapes[0]
    if self.is_informative:
        return processing_fn(transformed_tapes)
    for op in transformed_tape.circuit:
        qml.apply(op)
    mps = transformed_tape.measurements
    if not mps:
        return qfunc_output
    if isinstance(qfunc_output, qml.measurements.MeasurementProcess):
        return tuple(mps) if len(mps) > 1 else mps[0]
    if isinstance(qfunc_output, (tuple, list)):
        return type(qfunc_output)(mps)
    interface = qml.math.get_interface(qfunc_output)
    return qml.math.asarray(mps, like=interface)