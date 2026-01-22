from __future__ import annotations
import typing
import rustworkx as rx
from qiskit.pulse.channels import Channel
from qiskit.pulse.exceptions import UnassignedReferenceError
def block_to_dag(block: ScheduleBlock) -> rx.PyDAG:
    """Convert schedule block instruction into DAG.

    ``ScheduleBlock`` can be represented as a DAG as needed.
    For example, equality of two programs are efficiently checked on DAG representation.

    .. code-block:: python

        with pulse.build() as sched1:
            with pulse.align_left():
                pulse.play(my_gaussian0, pulse.DriveChannel(0))
                pulse.shift_phase(1.57, pulse.DriveChannel(2))
                pulse.play(my_gaussian1, pulse.DriveChannel(1))

        with pulse.build() as sched2:
            with pulse.align_left():
                pulse.shift_phase(1.57, pulse.DriveChannel(2))
                pulse.play(my_gaussian1, pulse.DriveChannel(1))
                pulse.play(my_gaussian0, pulse.DriveChannel(0))

    Here the ``sched1 `` and ``sched2`` are different implementations of the same program,
    but it is difficult to confirm on the list representation.

    Another example is instruction optimization.

    .. code-block:: python

        with pulse.build() as sched:
            with pulse.align_left():
                pulse.shift_phase(1.57, pulse.DriveChannel(1))
                pulse.play(my_gaussian0, pulse.DriveChannel(0))
                pulse.shift_phase(-1.57, pulse.DriveChannel(1))

    In above program two ``shift_phase`` instructions can be cancelled out because
    they are consecutive on the same drive channel.
    This can be easily found on the DAG representation.

    Args:
        block ("ScheduleBlock"): A schedule block to be converted.

    Returns:
        Instructions in DAG representation.

    Raises:
        PulseError: When the context is invalid subclass.
    """
    if block.alignment_context.is_sequential:
        return _sequential_allocation(block)
    return _parallel_allocation(block)