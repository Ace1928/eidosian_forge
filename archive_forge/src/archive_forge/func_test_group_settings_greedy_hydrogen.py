import cirq
def test_group_settings_greedy_hydrogen():
    qubits = cirq.LineQubit.range(4)
    q0, q1, q2, q3 = qubits
    terms = [0.1711977489805745 * cirq.Z(q0), 0.17119774898057447 * cirq.Z(q1), -0.2227859302428928 * cirq.Z(q2), -0.22278593024287646 * cirq.Z(q3), 0.16862219157249939 * cirq.Z(q0) * cirq.Z(q1), 0.04532220205777764 * cirq.Y(q0) * cirq.X(q1) * cirq.X(q2) * cirq.Y(q3), -0.0453222020577776 * cirq.Y(q0) * cirq.Y(q1) * cirq.X(q2) * cirq.X(q3), -0.0453222020577776 * cirq.X(q0) * cirq.X(q1) * cirq.Y(q2) * cirq.Y(q3), 0.04532220205777764 * cirq.X(q0) * cirq.Y(q1) * cirq.Y(q2) * cirq.X(q3), 0.12054482203290037 * cirq.Z(q0) * cirq.Z(q2), 0.16586702409067802 * cirq.Z(q0) * cirq.Z(q3), 0.16586702409067802 * cirq.Z(q1) * cirq.Z(q2), 0.12054482203290037 * cirq.Z(q1) * cirq.Z(q3), 0.1743484418396392 * cirq.Z(q2) * cirq.Z(q3)]
    settings = cirq.work.observables_to_settings(terms, qubits)
    grouped_settings = cirq.work.group_settings_greedy(settings)
    assert len(grouped_settings) == 5
    group_max_obs_should_be = [cirq.Y(q0) * cirq.X(q1) * cirq.X(q2) * cirq.Y(q3), cirq.Y(q0) * cirq.Y(q1) * cirq.X(q2) * cirq.X(q3), cirq.X(q0) * cirq.X(q1) * cirq.Y(q2) * cirq.Y(q3), cirq.X(q0) * cirq.Y(q1) * cirq.Y(q2) * cirq.X(q3), cirq.Z(q0) * cirq.Z(q1) * cirq.Z(q2) * cirq.Z(q3)]
    group_max_settings_should_be = cirq.work.observables_to_settings(group_max_obs_should_be, qubits)
    assert set(grouped_settings.keys()) == set(group_max_settings_should_be)
    groups = list(grouped_settings.values())
    assert len(groups[0]) == 1
    assert len(groups[1]) == 1
    assert len(groups[2]) == 1
    assert len(groups[3]) == 1
    assert len(groups[4]) == len(terms) - 4