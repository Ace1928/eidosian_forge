from typing import Any, Dict, Optional, Sequence, Type, Union, TYPE_CHECKING
from cirq import ops, protocols
from cirq.protocols.decompose_protocol import DecomposeResult
from cirq.transformers.analytical_decompositions import two_qubit_to_sqrt_iswap
from cirq.transformers.target_gatesets import compilation_target_gateset
Initializes `cirq.SqrtIswapTargetGateset`

        Args:
            atol: A limit on the amount of absolute error introduced by the decomposition.
            required_sqrt_iswap_count: When specified, the `decompose_to_target_gateset` will
                decompose each operation into exactly this many sqrt-iSWAP gates even if fewer is
                possible (maximum 3). A ValueError will be raised if this number is 2 or lower and
                synthesis of the operation requires more.
            use_sqrt_iswap_inv: If True, `cirq.SQRT_ISWAP_INV` is used as part of the gateset,
                instead of `cirq.SQRT_ISWAP`.
            additional_gates: Sequence of additional gates / gate families which should also
              be "accepted" by this gateset. This is empty by default.

        Raises:
            ValueError: If `required_sqrt_iswap_count` is specified and is not 0, 1, 2, or 3.
        