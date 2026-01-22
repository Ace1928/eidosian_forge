import itertools
import numpy as np
import pytest
import cirq
def test_lt_other_type():
    with pytest.raises(TypeError):
        _ = cirq.X < object()