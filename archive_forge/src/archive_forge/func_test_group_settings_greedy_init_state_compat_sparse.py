import cirq
def test_group_settings_greedy_init_state_compat_sparse():
    q0, q1 = cirq.LineQubit.range(2)
    settings = [cirq.work.InitObsSetting(init_state=cirq.KET_PLUS(q0), observable=cirq.X(q0)), cirq.work.InitObsSetting(init_state=cirq.KET_ZERO(q1), observable=cirq.Z(q1))]
    grouped_settings = cirq.work.group_settings_greedy(settings)
    grouped_settings_should_be = {cirq.work.InitObsSetting(init_state=cirq.KET_PLUS(q0) * cirq.KET_ZERO(q1), observable=cirq.X(q0) * cirq.Z(q1)): settings}
    assert grouped_settings == grouped_settings_should_be