import dataclasses
import cirq
import cirq_google
import pytest
from cirq_google import (
def test_bitstrings_measurement():
    bs = BitstringsMeasurement(n_repetitions=10000)
    cirq.testing.assert_equivalent_repr(bs, global_vals={'cirq_google': cirq_google})