from __future__ import annotations
import functools
from .instruction import Instruction
from .gate import Gate
from .controlledgate import ControlledGate, _ctrl_state_to_int
def stdlib_singleton_key(*, num_ctrl_qubits: int=0):
    """Create an implementation of the abstract method
    :meth:`SingletonInstruction._singleton_lookup_key`, for standard-library instructions whose
    ``__init__`` signatures match the one given here.

    .. warning::

        This method is not safe for use in classes defined outside of Qiskit; it is not included in
        the backwards compatibility guarantees.  This is because we guarantee that the call
        signatures of the base classes are backwards compatible in the sense that we will only
        replace them (without warning) contravariantly, but if you use this method, you effectively
        use the signature *invariantly*, and we cannot guarantee that.

    Args:
        num_ctrl_qubits: if given, this implies that the gate is a :class:`.ControlledGate`, and
            will have a fixed number of qubits that are used as the control.  This is necessary to
            allow ``ctrl_state`` to be given as either ``None`` or as an all-ones integer/string.
    """
    if num_ctrl_qubits:

        def key(label=None, ctrl_state=None, *, duration=None, unit='dt', _base_label=None):
            if label is None and duration is None and (unit == 'dt') and (_base_label is None):
                ctrl_state = _ctrl_state_to_int(ctrl_state, num_ctrl_qubits)
                return (ctrl_state,)
            return None
    else:

        def key(label=None, *, duration=None, unit='dt'):
            if label is None and duration is None and (unit == 'dt'):
                return ()
            return None
    return staticmethod(key)