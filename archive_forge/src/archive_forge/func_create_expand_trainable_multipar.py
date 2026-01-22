import contextlib
import pennylane as qml
from pennylane.operation import (
def create_expand_trainable_multipar(tape, use_tape_argnum=False):
    """Creates the expand_trainable_multipar expansion transform with an option to include argnums."""
    if not use_tape_argnum:
        return expand_trainable_multipar
    trainable_par_info = [tape._par_info[i] for i in tape.trainable_params]
    trainable_ops = [info['op'] for info in trainable_par_info]

    @qml.BooleanFn
    def _is_trainable(obj):
        return obj in trainable_ops
    return create_expand_fn(depth=10, stop_at=not_tape | is_measurement | has_nopar | ~_is_trainable | has_gen & ~gen_is_multi_term_hamiltonian, docstring=_expand_trainable_multipar_doc)