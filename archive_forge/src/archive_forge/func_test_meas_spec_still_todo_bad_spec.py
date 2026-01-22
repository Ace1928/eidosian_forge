import tempfile
from typing import Iterable, Dict, List
import numpy as np
import pytest
import cirq
import cirq.work as cw
from cirq.work import _MeasurementSpec, BitstringAccumulator, group_settings_greedy, InitObsSetting
from cirq.work.observable_measurement import (
def test_meas_spec_still_todo_bad_spec():
    bsa, meas_spec = _set_up_meas_specs_for_testing()

    class BadStopping(StoppingCriteria):

        def more_repetitions(self, accumulator: BitstringAccumulator) -> int:
            return -23
    bad_stop = BadStopping()
    with pytest.raises(ValueError, match='positive'):
        _, _ = _check_meas_specs_still_todo(meas_specs=[meas_spec], accumulators={meas_spec: bsa}, stopping_criteria=bad_stop)