import tempfile
from typing import Iterable, Dict, List
import numpy as np
import pytest
import cirq
import cirq.work as cw
from cirq.work import _MeasurementSpec, BitstringAccumulator, group_settings_greedy, InitObsSetting
from cirq.work.observable_measurement import (
def test_get_params_for_setting():
    qubits = cirq.LineQubit.range(3)
    a, b, c = qubits
    init_state = cirq.KET_PLUS(a) * cirq.KET_ZERO(b)
    observable = cirq.X(a) * cirq.Y(b)
    setting = cw.InitObsSetting(init_state=init_state, observable=observable)
    padded_setting = _pad_setting(setting, qubits=qubits)
    assert padded_setting.init_state == cirq.KET_PLUS(a) * cirq.KET_ZERO(b) * cirq.KET_ZERO(c)
    assert padded_setting.observable == cirq.X(a) * cirq.Y(b) * cirq.Z(c)
    assert init_state == cirq.KET_PLUS(a) * cirq.KET_ZERO(b)
    assert observable == cirq.X(a) * cirq.Y(b)
    needs_init_layer = True
    with pytest.raises(ValueError):
        _get_params_for_setting(padded_setting, flips=[0, 0], qubits=qubits, needs_init_layer=needs_init_layer)
    params = _get_params_for_setting(padded_setting, flips=[0, 0, 1], qubits=qubits, needs_init_layer=needs_init_layer)
    assert all((x in params for x in ['q(0)-Xf', 'q(0)-Yf', 'q(1)-Xf', 'q(1)-Yf', 'q(2)-Xf', 'q(2)-Yf', 'q(0)-Xi', 'q(0)-Yi', 'q(1)-Xi', 'q(1)-Yi', 'q(2)-Xi', 'q(2)-Yi']))
    circuit = cirq.Circuit(cirq.I.on_each(*qubits))
    circuit = _with_parameterized_layers(circuit, qubits=qubits, needs_init_layer=needs_init_layer)
    circuit = circuit[:-1]
    psi = cirq.Simulator().simulate(circuit, param_resolver=params)
    ma = cirq.Z(a).expectation_from_state_vector(psi.final_state_vector, qubit_map=psi.qubit_map)
    mb = cirq.Z(b).expectation_from_state_vector(psi.final_state_vector, qubit_map=psi.qubit_map)
    mc = cirq.Z(c).expectation_from_state_vector(psi.final_state_vector, qubit_map=psi.qubit_map)
    np.testing.assert_allclose([ma, mb, mc], [1, 0, -1])