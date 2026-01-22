from typing import List
import numpy as np
import pytest
import cirq
def test_merge_complex_circuit_preserving_moment_structure():
    q = cirq.LineQubit.range(3)
    c_orig = cirq.Circuit(cirq.Moment(cirq.H.on_each(*q)), cirq.CNOT(q[0], q[2]), cirq.CNOT(*q[0:2]), cirq.H(q[0]), cirq.CZ(*q[:2]), cirq.X(q[0]), cirq.Y(q[1]), cirq.CNOT(*q[0:2]), cirq.CNOT(*q[1:3]).with_tags('ignore'), cirq.X(q[0]), cirq.Moment(cirq.X(q[0]).with_tags('ignore'), cirq.Y(q[1]), cirq.Z(q[2])), cirq.Moment(cirq.CNOT(*q[:2]), cirq.measure(q[2], key='a')), cirq.X(q[0]).with_classical_controls('a'), strategy=cirq.InsertStrategy.NEW)
    cirq.testing.assert_has_diagram(c_orig, "\n0: ───H───@───@───H───@───X───────@─────────────────X───X['ignore']───@───X───\n          │   │       │           │                                   │   ║\n1: ───H───┼───X───────@───────Y───X───@['ignore']───────Y─────────────X───╫───\n          │                           │                                   ║\n2: ───H───X───────────────────────────X─────────────────Z─────────────M───╫───\n                                                                      ║   ║\na: ═══════════════════════════════════════════════════════════════════@═══^═══\n")
    component_id = 0

    def rewriter_merge_to_circuit_op(op: 'cirq.CircuitOperation') -> 'cirq.OP_TREE':
        nonlocal component_id
        component_id = component_id + 1
        return op.with_tags(f'{component_id}')
    c_new = cirq.merge_k_qubit_unitaries(c_orig, k=2, context=cirq.TransformerContext(tags_to_ignore=('ignore',)), rewriter=rewriter_merge_to_circuit_op)
    cirq.testing.assert_has_diagram(cirq.drop_empty_moments(c_new), "\n      [ 0: ───H───@─── ]        [ 0: ───────@───H───@───X───@───X─── ]                                            [ 0: ───────@─── ]\n0: ───[           │    ]────────[           │       │       │        ]──────────────────────X['ignore']───────────[           │    ]────────X───\n      [ 2: ───H───X─── ]['1']   [ 1: ───H───X───────@───Y───X─────── ]['2']                                       [ 1: ───Y───X─── ]['4']   ║\n      │                         │                                                                                 │                         ║\n1: ───┼─────────────────────────#2────────────────────────────────────────────@['ignore']─────────────────────────#2────────────────────────╫───\n      │                                                                       │                                                             ║\n2: ───#2──────────────────────────────────────────────────────────────────────X─────────────[ 2: ───Z─── ]['3']───M─────────────────────────╫───\n                                                                                                                  ║                         ║\na: ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════@═════════════════════════^═══")
    component_id = 0

    def rewriter_replace_with_decomp(op: 'cirq.CircuitOperation') -> 'cirq.OP_TREE':
        nonlocal component_id
        component_id = component_id + 1
        tag = f'{component_id}'
        if len(op.qubits) == 1:
            return [cirq.T(op.qubits[0]).with_tags(tag)]
        one_layer = [op.with_tags(tag) for op in cirq.T.on_each(*op.qubits)]
        two_layer = [cirq.SQRT_ISWAP(*op.qubits).with_tags(tag)]
        return [one_layer, two_layer, one_layer]
    c_new = cirq.merge_k_qubit_unitaries(c_orig, k=2, context=cirq.TransformerContext(tags_to_ignore=('ignore',)), rewriter=rewriter_replace_with_decomp)
    cirq.testing.assert_has_diagram(cirq.drop_empty_moments(c_new), "\n0: ───T['1']───iSwap['1']───T['1']───T['2']───iSwap['2']───T['2']─────────────────X['ignore']───T['4']───iSwap['4']───T['4']───X───\n               │                              │                                                          │                     ║\n1: ────────────┼─────────────────────T['2']───iSwap^0.5────T['2']───@['ignore']─────────────────T['4']───iSwap^0.5────T['4']───╫───\n               │                                                    │                                                          ║\n2: ───T['1']───iSwap^0.5────T['1']──────────────────────────────────X─────────────T['3']────────M──────────────────────────────╫───\n                                                                                                ║                              ║\na: ═════════════════════════════════════════════════════════════════════════════════════════════@══════════════════════════════^═══")