from typing import Optional, Callable, Hashable, Sequence, TYPE_CHECKING
from cirq import circuits
from cirq.protocols import decompose_protocol as dp
from cirq.transformers import transformer_api, transformer_primitives
Transforms the given circuit into an equivalent circuit using gates accepted by `gateset`.

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
    