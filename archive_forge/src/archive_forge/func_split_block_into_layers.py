from qiskit.circuit import QuantumCircuit, CircuitInstruction, ClassicalRegister
from qiskit.circuit.controlflow import condition_resources
from . import DAGOpNode, DAGCircuit, DAGDependency
from .exceptions import DAGCircuitError
def split_block_into_layers(block):
    """Splits a block of nodes into sub-blocks of non-overlapping instructions
    (or, in other words, into depth-1 sub-blocks).
    """
    bit_depths = {}
    layers = []
    for node in block:
        cur_bits = set(node.qargs)
        cur_bits.update(node.cargs)
        cond = getattr(node.op, 'condition', None)
        if cond is not None:
            cur_bits.update(condition_resources(cond).clbits)
        cur_depth = max((bit_depths.get(bit, 0) for bit in cur_bits))
        while len(layers) <= cur_depth:
            layers.append([])
        for bit in cur_bits:
            bit_depths[bit] = cur_depth + 1
        layers[cur_depth].append(node)
    return layers