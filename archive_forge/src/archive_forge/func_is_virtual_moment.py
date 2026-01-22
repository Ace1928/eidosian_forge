from typing import Any, Dict, Iterable, Sequence, TYPE_CHECKING, Union, Callable
from cirq import ops, protocols, value
from cirq._import import LazyLoader
from cirq._doc import document
def is_virtual_moment(self, moment: 'cirq.Moment') -> bool:
    """Returns true iff the given moment is non-empty and all of its
        operations are virtual.

        Moments for which this method returns True should not have additional
        noise applied to them.

        Args:
            moment: ``cirq.Moment`` to check for non-virtual operations.

        Returns:
            True if "moment" is non-empty and all operations in "moment" are
            virtual; false otherwise.
        """
    if not moment.operations:
        return False
    return all((ops.VirtualTag() in op.tags for op in moment))