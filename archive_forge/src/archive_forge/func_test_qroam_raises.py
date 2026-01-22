import cirq
import cirq_ft
import numpy as np
import pytest
from cirq_ft import infra
from cirq_ft.infra.bit_tools import iter_bits
from cirq_ft.deprecation import allow_deprecated_cirq_ft_use_in_tests
@allow_deprecated_cirq_ft_use_in_tests
def test_qroam_raises():
    with pytest.raises(ValueError, match='must be of equal length'):
        _ = cirq_ft.SelectSwapQROM([1, 2], [1, 2, 3])