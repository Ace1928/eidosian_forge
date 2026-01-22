import dataclasses
import datetime
import time
import numpy as np
import pytest
import cirq
import cirq.work as cw
from cirq.work.observable_measurement_data import (
from cirq.work.observable_settings import _MeasurementSpec
def test_readout_correction():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    ro_bsa, ro_settings, ro_meas_spec_setting = _get_mock_readout_calibration()
    assert ro_bsa.mean(ro_settings[0]) == 0.8
    assert ro_bsa.mean(ro_settings[1]) == 0.82
    assert np.isclose(ro_bsa.mean(ro_meas_spec_setting), 0.8 * 0.82, atol=0.05)
    bitstrings = np.array([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 1], [1, 1]], dtype=np.uint8)
    chunksizes = np.asarray([len(bitstrings)])
    timestamps = np.asarray([datetime.datetime.now()])
    qubit_to_index = {a: 0, b: 1}
    settings = list(cw.observables_to_settings([cirq.X(a) * cirq.Y(b), cirq.X(a), cirq.Y(b)], qubits=[a, b]))
    meas_spec = _MeasurementSpec(settings[0], {})
    bsa1 = cw.BitstringAccumulator(meas_spec=meas_spec, simul_settings=settings, qubit_to_index=qubit_to_index, bitstrings=bitstrings, chunksizes=chunksizes, timestamps=timestamps)
    np.testing.assert_allclose([1 - 1 / 4, 1 - 1 / 4, 1 - 2 / 4], bsa1.means())
    np.testing.assert_allclose([0.75, 0.75, 0.5], bsa1.means())
    bsa2 = cw.BitstringAccumulator(meas_spec=meas_spec, simul_settings=settings, qubit_to_index=qubit_to_index, bitstrings=bitstrings, chunksizes=chunksizes, timestamps=timestamps, readout_calibration=ro_bsa)
    for setting in settings:
        assert bsa2.variance(setting) > bsa1.variance(setting)
    np.testing.assert_allclose([0.75 / (0.8 * 0.82), 0.75 / 0.8, 0.5 / 0.82], bsa2.means(), atol=0.01)
    ro_bsa_50_50, _, _ = _get_mock_readout_calibration(qa_0=50, qa_1=50)
    bsa3 = cw.BitstringAccumulator(meas_spec=meas_spec, simul_settings=settings, qubit_to_index=qubit_to_index, bitstrings=bitstrings, chunksizes=chunksizes, timestamps=timestamps, readout_calibration=ro_bsa_50_50)
    with pytest.raises(ZeroDivisionError):
        bsa3.means()
    assert bsa3.variance(settings[1]) == np.inf