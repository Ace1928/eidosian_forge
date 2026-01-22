import pytest
import numpy as np
import cirq
import cirq_google
from cirq_google.transformers.target_gatesets import sycamore_gateset
def test_merge_swap_rzz_and_2q_unitaries_deep():
    q = cirq.LineQubit.range(3)
    swap_rzz = cirq.FrozenCircuit(cirq.SWAP(*q[:2]), cirq.ZZ(*q[:2]) ** 0.5)
    rzz_swap = cirq.FrozenCircuit(cirq.ZZ(*q[1:]) ** 0.25, cirq.SWAP(*q[1:]))
    x_cnot_x = cirq.FrozenCircuit(cirq.X(q[0]), cirq.CNOT(*q[:2]), cirq.X(q[0]))
    x_cz_x = cirq.FrozenCircuit(cirq.X(q[2]), cirq.CZ(*q[1:]), cirq.X(q[2]))
    c_orig = cirq.Circuit(cirq.CircuitOperation(swap_rzz).repeat(3).with_tags('ignore'), cirq.CircuitOperation(rzz_swap).repeat(5).with_tags('preserve_tag'), cirq.CircuitOperation(x_cnot_x).repeat(7).with_tags('ignore'), cirq.CircuitOperation(x_cz_x).repeat(9).with_tags('preserve_tag'), cirq.CircuitOperation(cirq.FrozenCircuit([swap_rzz, rzz_swap, x_cnot_x, x_cz_x], cirq.Moment((cirq.Y(qq).with_tags('ignore') for qq in q)))))
    t_swap_rzz = '_merged_swap_rzz_tag'
    t_2q_cmp = '_merged_2q_unitaries_component'
    t_all = '_intermediate_result_tag_apply_to_all'

    def _wrap_cop(c: cirq.FrozenCircuit, *tags) -> cirq.FrozenCircuit:
        return cirq.FrozenCircuit(cirq.CircuitOperation(c).with_tags(*tags, t_all))
    c_expected = cirq.Circuit(cirq.CircuitOperation(swap_rzz).repeat(3).with_tags('ignore'), cirq.CircuitOperation(_wrap_cop(rzz_swap, t_swap_rzz)).repeat(5).with_tags('preserve_tag'), cirq.CircuitOperation(x_cnot_x).repeat(7).with_tags('ignore'), cirq.CircuitOperation(_wrap_cop(x_cz_x, t_2q_cmp)).repeat(9).with_tags('preserve_tag'), cirq.CircuitOperation(cirq.FrozenCircuit([_wrap_cop(swap_rzz, t_swap_rzz), _wrap_cop(rzz_swap, t_swap_rzz)], [_wrap_cop(x_cnot_x, t_2q_cmp), _wrap_cop(x_cz_x, t_2q_cmp)], cirq.Moment((cirq.Y(qq).with_tags('ignore') for qq in q)))))
    context = cirq.TransformerContext(tags_to_ignore=['ignore'], deep=True)
    c_new = sycamore_gateset.merge_swap_rzz_and_2q_unitaries(c_orig, context=context, merged_swap_rzz_tag=t_swap_rzz, merged_2q_component_tag=t_2q_cmp, intermediate_result_tag=t_all)
    cirq.testing.assert_same_circuits(cirq.drop_empty_moments(c_new, context=context), c_expected)