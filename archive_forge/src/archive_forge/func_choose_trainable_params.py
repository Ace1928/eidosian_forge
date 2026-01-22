from functools import partial
import warnings
import pennylane as qml
from pennylane.transforms.tape_expand import expand_invalid_trainable
from pennylane.measurements import (
def choose_trainable_params(tape, argnum=None):
    """Returns a list of trainable parameter indices in the tape.

    Chooses the subset of trainable parameters to compute the Jacobian for. The function
    returns a list of indices with respect to the list of trainable parameters. If argnum
    is not provided, all trainable parameters are considered.

    Args:
        tape (`~.QuantumScript`): the tape to analyze
        argnum (int, list(int), None): Indices for trainable parameters(s)
            to compute the Jacobian for.

    Returns:
        list: list of the trainable parameter indices

    """
    if argnum is None:
        return [idx for idx, _ in enumerate(tape.trainable_params)]
    if isinstance(argnum, int):
        argnum = [argnum]
    if len(argnum) == 0:
        warnings.warn('No trainable parameters were specified for computing the Jacobian.', UserWarning)
    return argnum