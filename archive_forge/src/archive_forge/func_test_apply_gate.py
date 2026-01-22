import unittest.mock as mock
import numpy as np
import sympy
import cirq
def test_apply_gate():
    q0, q1 = cirq.LineQubit.range(2)
    state = mock.Mock()
    args = cirq.StabilizerSimulationState(state=state, qubits=[q0, q1])
    assert args._strat_apply_gate(cirq.X, [q0]) is True
    state.apply_x.assert_called_with(0, 1.0, 0.0)
    state.reset_mock()
    assert args._strat_apply_gate(cirq.X ** 2, [q0]) is True
    state.apply_x.assert_called_with(0, 2.0, 0.0)
    state.reset_mock()
    assert args._strat_apply_gate(cirq.X ** sympy.Symbol('t'), [q0]) is NotImplemented
    state.apply_x.assert_not_called()
    state.reset_mock()
    assert args._strat_apply_gate(cirq.XPowGate(exponent=2, global_shift=1.3), [q1]) is True
    state.apply_x.assert_called_with(1, 2.0, 1.3)
    state.reset_mock()
    assert args._strat_apply_gate(cirq.X ** 1.4, [q0]) == NotImplemented
    state.apply_x.assert_not_called()
    state.reset_mock()
    assert args._strat_apply_gate(cirq.Y, [q0]) is True
    state.apply_y.assert_called_with(0, 1.0, 0.0)
    state.reset_mock()
    assert args._strat_apply_gate(cirq.Z, [q0]) is True
    state.apply_z.assert_called_with(0, 1.0, 0.0)
    state.reset_mock()
    assert args._strat_apply_gate(cirq.H, [q0]) is True
    state.apply_h.assert_called_with(0, 1.0, 0.0)
    state.reset_mock()
    assert args._strat_apply_gate(cirq.CX, [q0, q1]) is True
    state.apply_cx.assert_called_with(0, 1, 1.0, 0.0)
    state.reset_mock()
    assert args._strat_apply_gate(cirq.CX, [q1, q0]) is True
    state.apply_cx.assert_called_with(1, 0, 1.0, 0.0)
    state.reset_mock()
    assert args._strat_apply_gate(cirq.CZ, [q0, q1]) is True
    state.apply_cz.assert_called_with(0, 1, 1.0, 0.0)
    state.reset_mock()
    assert args._strat_apply_gate(cirq.GlobalPhaseGate(1j), []) is True
    state.apply_global_phase.assert_called_with(1j)
    state.reset_mock()
    assert args._strat_apply_gate(cirq.GlobalPhaseGate(sympy.Symbol('t')), []) is NotImplemented
    state.apply_global_phase.assert_not_called()
    state.reset_mock()
    assert args._strat_apply_gate(cirq.SWAP, [q0, q1]) is True
    state.apply_cx.assert_has_calls([mock.call(0, 1), mock.call(1, 0, 1.0, 0.0), mock.call(0, 1)])
    state.reset_mock()
    assert args._strat_apply_gate(cirq.SwapPowGate(exponent=2, global_shift=1.3), [q0, q1]) is True
    state.apply_cx.assert_has_calls([mock.call(0, 1), mock.call(1, 0, 2.0, 1.3), mock.call(0, 1)])
    state.reset_mock()
    assert args._strat_apply_gate(cirq.BitFlipChannel(0.5), [q0]) == NotImplemented
    state.apply_x.assert_not_called()