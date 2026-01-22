from typing import Tuple
from numpy.typing import NDArray
import cirq
import cirq_ft
import numpy as np
import pytest
from cirq_ft import infra
from cirq._compat import cached_property
from cirq_ft.infra.bit_tools import iter_bits
from cirq_ft.deprecation import allow_deprecated_cirq_ft_use_in_tests
def rotation_ops(theta: int) -> cirq.OP_TREE:
    for i, b in enumerate(bin(theta)[2:][::-1]):
        if b == '1':
            yield cirq.pow(rotation_gate.on(*g.quregs['rotations_target']), 1 / 2 ** (1 + i))