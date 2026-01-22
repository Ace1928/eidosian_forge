import math
import heapq
from collections import OrderedDict, defaultdict
import rustworkx as rx
from qiskit.circuit.commutation_library import SessionCommutationChecker as scc
from qiskit.circuit.controlflow import condition_resources
from qiskit.circuit.quantumregister import QuantumRegister, Qubit
from qiskit.circuit.classicalregister import ClassicalRegister, Clbit
from qiskit.dagcircuit.exceptions import DAGDependencyError
from qiskit.dagcircuit.dagdepnode import DAGDepNode
def replace_block_with_op(self, node_block, op, wire_pos_map, cycle_check=True):
    """Replace a block of nodes with a single node.

        This is used to consolidate a block of DAGDepNodes into a single
        operation. A typical example is a block of CX and SWAP gates consolidated
        into a LinearFunction. This function is an adaptation of a similar
        function from DAGCircuit.

        It is important that such consolidation preserves commutativity assumptions
        present in DAGDependency. As an example, suppose that every node in a
        block [A, B, C, D] commutes with another node E. Let F be the consolidated
        node, F = A o B o C o D. Then F also commutes with E, and thus the result of
        replacing [A, B, C, D] by F results in a valid DAGDependency. That is, any
        deduction about commutativity in consolidated DAGDependency is correct.
        On the other hand, suppose that at least one of the nodes, say B, does not commute
        with E. Then the consolidated DAGDependency would imply that F does not commute
        with E. Even though F and E may actually commute, it is still safe to assume that
        they do not. That is, the current implementation of consolidation may lead to
        suboptimal but not to incorrect results.

        Args:
            node_block (List[DAGDepNode]): A list of dag nodes that represents the
                node block to be replaced
            op (qiskit.circuit.Operation): The operation to replace the
                block with
            wire_pos_map (Dict[~qiskit.circuit.Qubit, int]): The dictionary mapping the qarg to
                the position. This is necessary to reconstruct the qarg order
                over multiple gates in the combined single op node.
            cycle_check (bool): When set to True this method will check that
                replacing the provided ``node_block`` with a single node
                would introduce a cycle (which would invalidate the
                ``DAGDependency``) and will raise a ``DAGDependencyError`` if a cycle
                would be introduced. This checking comes with a run time
                penalty. If you can guarantee that your input ``node_block`` is
                a contiguous block and won't introduce a cycle when it's
                contracted to a single node, this can be set to ``False`` to
                improve the runtime performance of this method.
        Raises:
            DAGDependencyError: if ``cycle_check`` is set to ``True`` and replacing
                the specified block introduces a cycle or if ``node_block`` is
                empty.
        """
    block_qargs = set()
    block_cargs = set()
    block_ids = [x.node_id for x in node_block]
    if not node_block:
        raise DAGDependencyError("Can't replace an empty node_block")
    for nd in node_block:
        block_qargs |= set(nd.qargs)
        block_cargs |= set(nd.cargs)
        cond = getattr(nd.op, 'condition', None)
        if cond is not None:
            block_cargs.update(condition_resources(cond).clbits)
    new_node = self._create_op_node(op, qargs=sorted(block_qargs, key=lambda x: wire_pos_map[x]), cargs=sorted(block_cargs, key=lambda x: wire_pos_map[x]))
    try:
        new_node.node_id = self._multi_graph.contract_nodes(block_ids, new_node, check_cycle=cycle_check)
    except rx.DAGWouldCycle as ex:
        raise DAGDependencyError('Replacing the specified node block would introduce a cycle') from ex