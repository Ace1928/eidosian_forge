import numpy as np
import pytest
import cirq
def test_tableau_then():
    t1, t2, expected_t = _three_identical_table(1)
    assert expected_t == t1.then(t2)
    t1, t2, expected_t = _three_identical_table(1)
    _ = [_H(t, 0) for t in (t1, expected_t)]
    _ = [_H(t, 0) for t in (t2, expected_t)]
    assert expected_t == t1.then(t2)
    t1, t2, expected_t = _three_identical_table(1)
    _ = [_X(t, 0) for t in (t1, expected_t)]
    _ = [_S(t, 0) for t in (t2, expected_t)]
    assert expected_t == t1.then(t2)
    assert expected_t != t2.then(t1)
    t1, t2, expected_t = _three_identical_table(1)
    _ = [_X(t, 0) for t in (t1, expected_t)]
    _ = [_H(t, 0) for t in (t1, expected_t)]
    _ = [_Z(t, 0) for t in (t1, expected_t)]
    _ = [_S(t, 0) for t in (t2, expected_t)]
    _ = [_H(t, 0) for t in (t2, expected_t)]
    assert expected_t == t1.then(t2)
    assert expected_t != t2.then(t1)
    t1, t2, expected_t = _three_identical_table(2)
    _ = [_H(t, 0) for t in (t1, expected_t)]
    _ = [_H(t, 1) for t in (t1, expected_t)]
    _ = [_H(t, 0) for t in (t2, expected_t)]
    _ = [_H(t, 1) for t in (t2, expected_t)]
    assert expected_t == t1.then(t2)
    t1, t2, expected_t = _three_identical_table(2)
    _ = [_H(t, 0) for t in (t1, expected_t)]
    _ = [_CNOT(t, 0, 1) for t in (t1, expected_t)]
    _ = [_S(t, 0) for t in (t2, expected_t)]
    _ = [_X(t, 1) for t in (t2, expected_t)]
    assert expected_t == t1.then(t2)
    assert expected_t != t2.then(t1)
    t1, t2, expected_t = _three_identical_table(2)
    _ = [_H(t, 0) for t in (t1, expected_t)]
    _ = [_CNOT(t, 0, 1) for t in (t1, expected_t)]
    _ = [_S(t, 1) for t in (t2, expected_t)]
    _ = [_CNOT(t, 1, 0) for t in (t2, expected_t)]
    assert expected_t == t1.then(t2)
    assert expected_t != t2.then(t1)

    def random_circuit(num_ops, num_qubits, seed=12345):
        prng = np.random.RandomState(seed)
        candidate_op = [_H, _S, _X, _Z]
        if num_qubits > 1:
            candidate_op = [_H, _S, _X, _Z, _CNOT]
        seq_op = []
        for _ in range(num_ops):
            op = prng.randint(len(candidate_op))
            if op != 4:
                args = (prng.randint(num_qubits),)
            else:
                args = prng.choice(num_qubits, 2, replace=False)
            seq_op.append((candidate_op[op], args))
        return seq_op
    for seed in range(100):
        t1, t2, expected_t = _three_identical_table(8)
        seq_op = random_circuit(num_ops=20, num_qubits=8, seed=seed)
        for i, (op, args) in enumerate(seq_op):
            if i < 7:
                _ = [op(t, *args) for t in (t1, expected_t)]
            else:
                _ = [op(t, *args) for t in (t2, expected_t)]
        assert expected_t == t1.then(t2)
    t1, t2, expected_t = _three_identical_table(100)
    seq_op = random_circuit(num_ops=1000, num_qubits=100)
    for i, (op, args) in enumerate(seq_op):
        if i < 350:
            _ = [op(t, *args) for t in (t1, expected_t)]
        else:
            _ = [op(t, *args) for t in (t2, expected_t)]
    assert expected_t == t1.then(t2)