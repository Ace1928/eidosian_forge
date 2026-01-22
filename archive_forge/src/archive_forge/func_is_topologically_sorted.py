import operator
import random
from typing import Any, Callable, cast, Iterable, TYPE_CHECKING
import networkx
from cirq import ops
def is_topologically_sorted(dag: 'cirq.contrib.CircuitDag', operations: 'cirq.OP_TREE', equals: Callable[[ops.Operation, ops.Operation], bool]=operator.eq) -> bool:
    """Whether a given order of operations is consistent with the DAG.

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
    """
    remaining_dag = dag.copy()
    frontier = [node for node in remaining_dag.nodes() if not remaining_dag.pred[node]]
    for operation in cast(Iterable[ops.Operation], ops.flatten_op_tree(operations)):
        for i, node in enumerate(frontier):
            if equals(node.val, operation):
                frontier.pop(i)
                succ = remaining_dag.succ[node]
                remaining_dag.remove_node(node)
                frontier.extend((new_node for new_node in succ if not remaining_dag.pred[new_node]))
                break
        else:
            return False
    return not bool(frontier)