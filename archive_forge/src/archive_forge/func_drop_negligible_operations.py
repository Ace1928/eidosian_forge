from typing import Optional, TYPE_CHECKING
from cirq import protocols
from cirq.transformers import transformer_api, transformer_primitives
@transformer_api.transformer
def drop_negligible_operations(circuit: 'cirq.AbstractCircuit', *, context: Optional['cirq.TransformerContext']=None, atol: float=1e-08) -> 'cirq.Circuit':
    """Removes operations with tiny effects.

    An operation `op` is considered to have a tiny effect if
    `cirq.trace_distance_bound(op) <= atol`.

    Args:
          circuit: Input circuit to transform.
          context: `cirq.TransformerContext` storing common configurable options for transformers.
          atol: Absolute tolerance to determine if an operation `op` is negligible --
                i.e. if `cirq.trace_distance_bound(op) <= atol`.

    Returns:
          Copy of the transformed input circuit.
    """
    if context is None:
        context = transformer_api.TransformerContext()

    def map_func(op: 'cirq.Operation', _: int) -> 'cirq.OP_TREE':
        return op if protocols.num_qubits(op) > 10 or protocols.is_measurement(op) or protocols.trace_distance_bound(op) > atol else []
    return transformer_primitives.map_operations(circuit, map_func, tags_to_ignore=context.tags_to_ignore, deep=context.deep).unfreeze(copy=False)