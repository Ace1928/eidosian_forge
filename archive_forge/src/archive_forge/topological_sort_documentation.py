import operator
import random
from typing import Any, Callable, cast, Iterable, TYPE_CHECKING
import networkx
from cirq import ops
Whether a given order of operations is consistent with the DAG.

    For example, suppose the (transitive reduction of the) circuit DAG is

         ╭─> Op2 ─╮
    Op1 ─┤        ├─> Op4
         ╰─> Op3 ─╯

    Then [Op1, Op2, Op3, Op4] and [Op1, Op3, Op2, Op4] (and any operations
    tree that flattens to one of them) are topologically sorted according
    to the DAG, and any other ordering of the four operations is not.

    Evaluates to False when the set of operations is different from those
    in the nodes of the DAG, regardless of the ordering.

    Args:
        dag: The circuit DAG.
        operations: The ordered operations.
        equals: The function to determine equality of operations. Defaults to
            `operator.eq`.

    Returns:
        Whether or not the operations given are topologically sorted
        according to the DAG.
    