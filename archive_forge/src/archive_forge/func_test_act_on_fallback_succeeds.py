from typing import Any, Sequence, Tuple
from typing_extensions import Self
import numpy as np
import pytest
import cirq
def test_act_on_fallback_succeeds():
    state = ExampleSimulationState(fallback_result=True)
    cirq.act_on(op, state)