from typing import Collection, Optional, Sequence, Tuple, Union
from numpy.typing import NDArray
import attr
import cirq
from cirq._compat import cached_property
from cirq_ft import infra
from cirq_ft.algos import reflection_using_prepare as rup
from cirq_ft.algos import select_and_prepare as sp
from cirq_ft.algos.mean_estimation import complex_phase_oracle
@cached_property
def selection_registers(self) -> Tuple[infra.SelectionRegister, ...]:
    return self.code.encoder.selection_registers