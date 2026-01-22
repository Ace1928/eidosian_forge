from typing import Sequence
import pytest
import numpy as np
import cirq
def test_single_qubit_readout_result_repr():
    result = cirq.experiments.SingleQubitReadoutCalibrationResult(zero_state_errors={cirq.LineQubit(0): 0.1}, one_state_errors={cirq.LineQubit(0): 0.2}, repetitions=1000, timestamp=0.3)
    cirq.testing.assert_equivalent_repr(result)