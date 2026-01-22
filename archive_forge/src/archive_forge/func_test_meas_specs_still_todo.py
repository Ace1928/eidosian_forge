import tempfile
from typing import Iterable, Dict, List
import numpy as np
import pytest
import cirq
import cirq.work as cw
from cirq.work import _MeasurementSpec, BitstringAccumulator, group_settings_greedy, InitObsSetting
from cirq.work.observable_measurement import (
def test_meas_specs_still_todo():
    bsa, meas_spec = _set_up_meas_specs_for_testing()
    stop = cw.RepetitionsStoppingCriteria(1000)
    still_todo, reps = _check_meas_specs_still_todo(meas_specs=[meas_spec], accumulators={meas_spec: bsa}, stopping_criteria=stop)
    assert still_todo == [meas_spec]
    assert reps == 1000
    bsa.consume_results(np.zeros((997, 3), dtype=np.uint8))
    still_todo, reps = _check_meas_specs_still_todo(meas_specs=[meas_spec], accumulators={meas_spec: bsa}, stopping_criteria=stop)
    assert still_todo == [meas_spec]
    assert reps == 3
    bsa.consume_results(np.zeros((reps, 3), dtype=np.uint8))
    still_todo, reps = _check_meas_specs_still_todo(meas_specs=[meas_spec], accumulators={meas_spec: bsa}, stopping_criteria=stop)
    assert still_todo == []
    assert reps == 0