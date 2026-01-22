from typing import Collection, Optional, Sequence, Tuple, Union
from numpy.typing import NDArray
import attr
import cirq
import numpy as np
from cirq._compat import cached_property
from cirq_ft import infra
from cirq_ft.algos import and_gate, apply_gate_to_lth_target, arithmetic_gates
from cirq_ft.algos import prepare_uniform_superposition as prep_u
from cirq_ft.algos import (
@cached_property
def junk_registers(self) -> Tuple[infra.Register, ...]:
    return (infra.Register('temp', 2),)