from typing import AbstractSet, Iterator, Any
import pytest
import numpy as np
import sympy
import cirq
def test_tagged_operation_forwards_protocols():
    """The results of all protocols applied to an operation with a tag should
    be equivalent to the result without tags.
    """
    q1 = cirq.GridQubit(1, 1)
    q2 = cirq.GridQubit(1, 2)
    h = cirq.H(q1)
    tag = 'tag1'
    tagged_h = cirq.H(q1).with_tags(tag)
    np.testing.assert_equal(cirq.unitary(tagged_h), cirq.unitary(h))
    assert cirq.has_unitary(tagged_h)
    assert cirq.decompose(tagged_h) == cirq.decompose(h)
    assert [*tagged_h._decompose_()] == cirq.decompose(h)
    assert cirq.pauli_expansion(tagged_h) == cirq.pauli_expansion(h)
    assert cirq.equal_up_to_global_phase(h, tagged_h)
    assert np.isclose(cirq.kraus(h), cirq.kraus(tagged_h)).all()
    assert cirq.measurement_key_name(cirq.measure(q1, key='blah').with_tags(tag)) == 'blah'
    assert cirq.measurement_key_obj(cirq.measure(q1, key='blah').with_tags(tag)) == cirq.MeasurementKey('blah')
    parameterized_op = cirq.XPowGate(exponent=sympy.Symbol('t'))(q1).with_tags(tag)
    assert cirq.is_parameterized(parameterized_op)
    resolver = cirq.study.ParamResolver({'t': 0.25})
    assert cirq.resolve_parameters(parameterized_op, resolver) == cirq.XPowGate(exponent=0.25)(q1).with_tags(tag)
    assert cirq.resolve_parameters_once(parameterized_op, resolver) == cirq.XPowGate(exponent=0.25)(q1).with_tags(tag)
    assert parameterized_op._unitary_() is NotImplemented
    assert parameterized_op._mixture_() is NotImplemented
    assert parameterized_op._kraus_() is NotImplemented
    y = cirq.Y(q1)
    tagged_y = cirq.Y(q1).with_tags(tag)
    assert tagged_y ** 0.5 == cirq.YPowGate(exponent=0.5)(q1)
    assert tagged_y * 2 == y * 2
    assert 3 * tagged_y == 3 * y
    assert cirq.phase_by(y, 0.125, 0) == cirq.phase_by(tagged_y, 0.125, 0)
    controlled_y = tagged_y.controlled_by(q2)
    assert controlled_y.qubits == (q2, q1)
    assert isinstance(controlled_y, cirq.Operation)
    assert not isinstance(controlled_y, cirq.TaggedOperation)
    classically_controlled_y = tagged_y.with_classical_controls('a')
    assert classically_controlled_y == y.with_classical_controls('a')
    assert isinstance(classically_controlled_y, cirq.Operation)
    assert not isinstance(classically_controlled_y, cirq.TaggedOperation)
    clifford_x = cirq.SingleQubitCliffordGate.X(q1)
    tagged_x = cirq.SingleQubitCliffordGate.X(q1).with_tags(tag)
    assert cirq.commutes(clifford_x, clifford_x)
    assert cirq.commutes(tagged_x, clifford_x)
    assert cirq.commutes(clifford_x, tagged_x)
    assert cirq.commutes(tagged_x, tagged_x)
    assert cirq.trace_distance_bound(y ** 0.001) == cirq.trace_distance_bound((y ** 0.001).with_tags(tag))
    flip = cirq.bit_flip(0.5)(q1)
    tagged_flip = cirq.bit_flip(0.5)(q1).with_tags(tag)
    assert cirq.has_mixture(tagged_flip)
    assert cirq.has_kraus(tagged_flip)
    flip_mixture = cirq.mixture(flip)
    tagged_mixture = cirq.mixture(tagged_flip)
    assert len(tagged_mixture) == 2
    assert len(tagged_mixture[0]) == 2
    assert len(tagged_mixture[1]) == 2
    assert tagged_mixture[0][0] == flip_mixture[0][0]
    assert np.isclose(tagged_mixture[0][1], flip_mixture[0][1]).all()
    assert tagged_mixture[1][0] == flip_mixture[1][0]
    assert np.isclose(tagged_mixture[1][1], flip_mixture[1][1]).all()
    qubit_map = {q1: 'q1'}
    qasm_args = cirq.QasmArgs(qubit_id_map=qubit_map)
    assert cirq.qasm(h, args=qasm_args) == cirq.qasm(tagged_h, args=qasm_args)
    cirq.testing.assert_has_consistent_apply_unitary(tagged_h)