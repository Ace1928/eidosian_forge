import itertools
import random
from typing import List, Tuple
import cirq
import cirq_ft
import numpy as np
import pytest
from cirq_ft import infra
from cirq_ft.infra.jupyter_tools import execute_notebook
from cirq_ft.deprecation import allow_deprecated_cirq_ft_use_in_tests
@allow_deprecated_cirq_ft_use_in_tests
def test_and_gate_raises():
    with pytest.raises(ValueError, match='at-least 2 control values'):
        _ = cirq_ft.And(cv=(1,))