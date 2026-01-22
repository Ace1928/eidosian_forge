import pytest
import numpy as np
import google.protobuf.text_format as text_format
import cirq
import cirq_google as cg
import cirq_google.api.v2 as v2
import cirq_google.engine.virtual_engine_factory as factory
def test_median_weber_device():
    q0, q1 = cirq.GridQubit.rect(1, 2, 1, 4)
    cal = factory.load_median_device_calibration('weber')
    assert np.isclose(cal['parallel_p00_error'][q0,][0], 0.00592)
    assert np.isclose(cal['single_qubit_readout_separation_error'][q1,][0], 0.0005456228122027591)
    assert np.isclose(cal['two_qubit_sycamore_gate_xeb_average_error_per_cycle'][q0, q1][0], 0.010057137642549896)