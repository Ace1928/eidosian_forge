from qiskit.transpiler.basepasses import AnalysisPass
from qiskit.circuit import Gate
from qiskit.dagcircuit import DAGOpNode, DAGInNode
def collect_key(x):
    """special key function for topological ordering.
            Heuristic for this is to push all gates involving measurement
            or barriers, etc. as far back as possible (because they force
            blocks to end). After that, we process gates in order of lowest
            number of qubits acted on to largest number of qubits acted on
            because these have less chance of increasing the size of blocks
            The key also processes all the non operation notes first so that
            input nodes do not mess with the top sort of op nodes
            """
    if isinstance(x, DAGInNode):
        return 'a'
    if not isinstance(x, DAGOpNode):
        return 'd'
    if isinstance(x.op, Gate):
        if x.op.is_parameterized() or getattr(x.op, 'condition', None) is not None:
            return 'c'
        return 'b' + chr(ord('a') + len(x.qargs))
    return 'd'