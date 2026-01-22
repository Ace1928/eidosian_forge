import tempfile
from typing import Iterable, Dict, List
import numpy as np
import pytest
import cirq
import cirq.work as cw
from cirq.work import _MeasurementSpec, BitstringAccumulator, group_settings_greedy, InitObsSetting
from cirq.work.observable_measurement import (
def test_meas_spec_still_todo_lots_of_params(monkeypatch):
    monkeypatch.setattr(cw.observable_measurement, 'MAX_REPETITIONS_PER_JOB', 30000)
    bsa, meas_spec = _set_up_meas_specs_for_testing()
    lots_of_meas_spec = [meas_spec] * 4
    stop = cw.RepetitionsStoppingCriteria(10000)
    with pytest.warns(UserWarning, match='will be throttled from 10000 to 7500'):
        _, _ = _check_meas_specs_still_todo(meas_specs=lots_of_meas_spec, accumulators={meas_spec: bsa}, stopping_criteria=stop)