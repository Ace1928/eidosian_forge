import tempfile
from typing import Iterable, Dict, List
import numpy as np
import pytest
import cirq
import cirq.work as cw
from cirq.work import _MeasurementSpec, BitstringAccumulator, group_settings_greedy, InitObsSetting
from cirq.work.observable_measurement import (
@pytest.mark.parametrize(['circuit', 'observable'], [(cirq.Circuit(cirq.X(Q) ** 0.2), cirq.Z(Q)), (cirq.Circuit(cirq.X(Q) ** (-0.5), cirq.Z(Q) ** 0.2), cirq.Y(Q)), (cirq.Circuit(cirq.Y(Q) ** 0.5, cirq.Z(Q) ** 0.2), cirq.X(Q))])
def test_XYZ_point8(circuit, observable):
    df = measure_observables_df(circuit, [observable], cirq.Simulator(seed=52), stopping_criteria=VarianceStoppingCriteria(0.001 ** 2))
    assert len(df) == 1, 'one observable'
    mean = df.loc[0]['mean']
    np.testing.assert_allclose(0.8, mean, atol=0.01)