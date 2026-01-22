from typing import Iterable, Sequence, Union
import attr
import cirq
import numpy as np
from cirq_ft import infra
from cirq_ft.deprecation import deprecated_cirq_ft_class
def registers(self) -> Sequence[Union[int, Sequence[int]]]:
    return ((2,) * self.selection_bitsize, (2,), (2,) * self.target_bitsize)