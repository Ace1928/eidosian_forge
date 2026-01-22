from typing import Collection, Optional, Sequence, Tuple, Union
from numpy.typing import NDArray
import attr
import cirq
from cirq._compat import cached_property
from cirq_ft import infra
from cirq_ft.algos import reflection_using_prepare as rup
from cirq_ft.algos import select_and_prepare as sp
from cirq_ft.algos.mean_estimation import complex_phase_oracle
Mean estimation operator $U=REFL_{p} ROT_{y}$ as per Sec 3.1 of arxiv.org:2208.07544.

    The MeanEstimationOperator (aka KO Operator) expects `CodeForRandomVariable` to specify the
    synthesizer and encoder, that follows LCU SELECT/PREPARE API for convenience. It is composed
    of two unitaries:

        - REFL_{p}: Reflection around the state prepared by synthesizer $P$. It applies the unitary
            $P^{\dagger}(2|0><0| - I)P$.
        - ROT_{y}: Applies a complex phase $\exp(i * -2\arctan{y_{w}})$ when the selection register
            stores $w$. This is achieved by using the encoder to encode $y(w)$ in a temporary target
            register.

    Note that both $REFL_{p}$ and $ROT_{y}$ only act upon a selection register, thus mean estimation
    operator expects only a selection register (and a control register, for a controlled version for
    phase estimation).
    