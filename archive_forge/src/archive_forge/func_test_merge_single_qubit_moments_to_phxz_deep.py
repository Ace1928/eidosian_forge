from typing import List
import cirq
def test_merge_single_qubit_moments_to_phxz_deep():
    q = cirq.LineQubit.range(3)
    x_t_y = cirq.FrozenCircuit(cirq.Moment(cirq.X.on_each(*q[:2])), cirq.Moment(cirq.T.on_each(*q[1:])), cirq.Moment(cirq.Y.on_each(*q[:2])))
    c_nested = cirq.FrozenCircuit(x_t_y, cirq.Moment(cirq.CZ(*q[:2]), cirq.Y(q[2])), x_t_y, cirq.Moment(cirq.Y(q[0]).with_tags('ignore'), cirq.Z.on_each(*q[1:])))
    c_nested_merged = cirq.FrozenCircuit([_phxz(-0.25, 0.0, 0.75)(q[1]), _phxz(0.25, 0.0, 0.25)(q[2]), _phxz(-0.5, 0.0, -1.0)(q[0])], [cirq.CZ(q[0], q[1]), cirq.Y(q[2])], [_phxz(-0.25, 0.0, 0.75)(q[1]), _phxz(0.25, 0.0, 0.25)(q[2]), _phxz(-0.5, 0.0, -1.0)(q[0])], cirq.Moment(cirq.Y(q[0]).with_tags('ignore'), cirq.Z.on_each(*q[1:])))
    c_orig = cirq.Circuit(c_nested, cirq.CircuitOperation(c_nested).repeat(4).with_tags('ignore'), c_nested, cirq.CircuitOperation(c_nested).repeat(5).with_tags('preserve_tags'), c_nested, cirq.CircuitOperation(c_nested).repeat(6))
    c_expected = cirq.Circuit(c_nested_merged, cirq.CircuitOperation(c_nested).repeat(4).with_tags('ignore'), c_nested_merged, cirq.CircuitOperation(c_nested_merged).repeat(5).with_tags('preserve_tags'), c_nested_merged, cirq.CircuitOperation(c_nested_merged).repeat(6))
    context = cirq.TransformerContext(tags_to_ignore=['ignore'], deep=True)
    c_new = cirq.merge_single_qubit_moments_to_phxz(c_orig, context=context)
    cirq.testing.assert_allclose_up_to_global_phase(c_new.unitary(), c_expected.unitary(), atol=1e-07)