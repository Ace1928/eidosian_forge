from typing import Collection, Optional, Sequence, Tuple, Union
from numpy.typing import NDArray
import attr
import cirq
from cirq._compat import cached_property
from cirq_ft import infra
from cirq_ft.algos import multi_control_multi_target_pauli as mcmt
from cirq_ft.algos import select_and_prepare
Applies reflection around a state prepared by `prepare_gate`

    Applies $R_{s} = I - 2|s><s|$ using $R_{s} = P^†(I - 2|0><0|)P$ s.t. $P|0> = |s>$.
    Here
        $|s>$: The state along which we want to reflect.
        $P$: Unitary that prepares that state $|s>$ from the zero state $|0>$
        $R_{s}$: Reflection operator that adds a `-1` phase to all states in the subspace
            spanned by $|s>$.

    The composite gate corresponds to implementing the following circuit:

    |control> ------------------ Z -------------------
                                 |
    |L>       ---- PREPARE^† --- o --- PREPARE -------


    Args:
        prepare_gate: An instance of `cq.StatePreparationAliasSampling` gate the corresponds to
            `PREPARE`.
        control_val: If 0/1, a controlled version of the reflection operator is constructed.
            Defaults to None, in which case the resulting reflection operator is not controlled.

    References:
        [Encoding Electronic Spectra in Quantum Circuits with Linear T Complexity]
        (https://arxiv.org/abs/1805.03662).
            Babbush et. al. (2018). Figure 1.
    