from typing import Optional, Callable, Hashable, Sequence, TYPE_CHECKING
from cirq import circuits
from cirq.protocols import decompose_protocol as dp
from cirq.transformers import transformer_api, transformer_primitives
@transformer_api.transformer
def optimize_for_target_gateset(circuit: 'cirq.AbstractCircuit', *, context: Optional['cirq.TransformerContext']=None, gateset: Optional['cirq.CompilationTargetGateset']=None, ignore_failures: bool=True) -> 'cirq.Circuit':
    """Transforms the given circuit into an equivalent circuit using gates accepted by `gateset`.

    1. Run all `gateset.preprocess_transformers`
    2. Convert operations using built-in cirq decompose + `gateset.decompose_to_target_gateset`.
    3. Run all `gateset.postprocess_transformers`

    Args:
        circuit: Input circuit to transform. It will not be modified.
        context: `cirq.TransformerContext` storing common configurable options for transformers.
        gateset: Target gateset, which should be an instance of `cirq.CompilationTargetGateset`.
        ignore_failures: If set, operations that fail to convert are left unchanged. If not set,
            conversion failures raise a ValueError.

    Returns:
        An equivalent circuit containing gates accepted by `gateset`.

    Raises:
        ValueError: If any input operation fails to convert and `ignore_failures` is False.
    """
    if gateset is None:
        return _decompose_operations_to_target_gateset(circuit, context=context, ignore_failures=ignore_failures)
    for transformer in gateset.preprocess_transformers:
        circuit = transformer(circuit, context=context)
    circuit = _decompose_operations_to_target_gateset(circuit, context=context, gateset=gateset, decomposer=gateset.decompose_to_target_gateset, ignore_failures=ignore_failures, tags_to_decompose=(gateset._intermediate_result_tag,))
    for transformer in gateset.postprocess_transformers:
        circuit = transformer(circuit, context=context)
    return circuit.unfreeze(copy=False)