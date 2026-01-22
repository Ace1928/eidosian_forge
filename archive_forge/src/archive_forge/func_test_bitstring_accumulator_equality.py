import dataclasses
import datetime
import time
import numpy as np
import pytest
import cirq
import cirq.work as cw
from cirq.work.observable_measurement_data import (
from cirq.work.observable_settings import _MeasurementSpec
def test_bitstring_accumulator_equality():
    et = cirq.testing.EqualsTester()
    bitstrings = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.uint8)
    chunksizes = np.asarray([4])
    timestamps = np.asarray([datetime.datetime.now()])
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    qubit_to_index = {a: 0, b: 1}
    obs = cirq.Z(a) * cirq.Z(b) * 10
    setting = cw.InitObsSetting(init_state=cirq.Z(a) * cirq.Z(b), observable=obs)
    meas_spec = _MeasurementSpec(setting, {})
    cirq.testing.assert_equivalent_repr(cw.BitstringAccumulator(meas_spec=meas_spec, simul_settings=[setting], qubit_to_index=qubit_to_index, bitstrings=bitstrings.copy(), chunksizes=chunksizes.copy(), timestamps=timestamps.copy()))
    et.add_equality_group(cw.BitstringAccumulator(meas_spec=meas_spec, simul_settings=[setting], qubit_to_index=qubit_to_index, bitstrings=bitstrings.copy(), chunksizes=chunksizes.copy(), timestamps=timestamps.copy()), cw.BitstringAccumulator(meas_spec=meas_spec, simul_settings=[setting], qubit_to_index=qubit_to_index, bitstrings=bitstrings.copy(), chunksizes=chunksizes.copy(), timestamps=timestamps.copy()))
    time.sleep(1)
    timestamps = np.asarray([datetime.datetime.now()])
    et.add_equality_group(cw.BitstringAccumulator(meas_spec=meas_spec, simul_settings=[setting], qubit_to_index=qubit_to_index, bitstrings=bitstrings, chunksizes=chunksizes, timestamps=timestamps))
    et.add_equality_group(cw.BitstringAccumulator(meas_spec=_MeasurementSpec(setting, {'a': 2}), simul_settings=[setting], qubit_to_index=qubit_to_index, bitstrings=bitstrings, chunksizes=chunksizes, timestamps=timestamps))
    bitstrings = bitstrings.copy()
    bitstrings[0] = [1, 1]
    et.add_equality_group(cw.BitstringAccumulator(meas_spec=meas_spec, simul_settings=[setting], qubit_to_index=qubit_to_index, bitstrings=bitstrings, chunksizes=chunksizes, timestamps=timestamps))
    chunksizes = np.asarray([2, 2])
    timestamps = np.asarray(list(timestamps) * 2)
    et.add_equality_group(cw.BitstringAccumulator(meas_spec=meas_spec, simul_settings=[setting], qubit_to_index=qubit_to_index, bitstrings=bitstrings, chunksizes=chunksizes, timestamps=timestamps))