from functools import partial
import warnings
import pennylane as qml
from pennylane.transforms.tape_expand import expand_invalid_trainable
from pennylane.measurements import (
def assert_no_tape_batching(tape, transform_name):
    """Check whether a tape is broadcasted and raise an error if this is the case.

    Args:
        tape (`~.QuantumScript`): measurements to analyze
        transform_name (str): Name of the gradient transform that queries the tape
    """
    if tape.batch_size is not None:
        raise NotImplementedError(f'Computing the gradient of broadcasted tapes with the {transform_name} gradient transform is currently not supported. See #4462 for details.')