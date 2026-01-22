import cirq
import cirq_ft
import numpy as np
import pytest
from cirq_ft.infra.jupyter_tools import execute_notebook
from cirq_ft.infra import split_qubits, merge_qubits, get_named_qubits
from cirq_ft.deprecation import allow_deprecated_cirq_ft_use_in_tests
@allow_deprecated_cirq_ft_use_in_tests
def test_registers():
    r1 = cirq_ft.Register('r1', 5, side=cirq_ft.infra.Side.LEFT)
    r2 = cirq_ft.Register('r2', 2, side=cirq_ft.infra.Side.RIGHT)
    r3 = cirq_ft.Register('r3', 1)
    regs = cirq_ft.Signature([r1, r2, r3])
    assert len(regs) == 3
    cirq.testing.assert_equivalent_repr(regs, setup_code='import cirq_ft')
    with pytest.raises(ValueError, match='unique'):
        _ = cirq_ft.Signature([r1, r1])
    assert regs[0] == r1
    assert regs[1] == r2
    assert regs[2] == r3
    assert regs[0:1] == tuple([r1])
    assert regs[0:2] == tuple([r1, r2])
    assert regs[1:3] == tuple([r2, r3])
    assert regs.get_left('r1') == r1
    assert regs.get_right('r2') == r2
    assert regs.get_left('r3') == r3
    assert r1 in regs
    assert r2 in regs
    assert r3 in regs
    assert list(regs) == [r1, r2, r3]
    qubits = cirq.LineQubit.range(8)
    qregs = split_qubits(regs, qubits)
    assert qregs['r1'].tolist() == cirq.LineQubit.range(5)
    assert qregs['r2'].tolist() == cirq.LineQubit.range(5, 5 + 2)
    assert qregs['r3'].tolist() == [cirq.LineQubit(7)]
    qubits = qubits[::-1]
    with pytest.raises(ValueError, match='qubit registers must be present'):
        _ = merge_qubits(regs, r1=qubits[:5], r2=qubits[5:7], r4=qubits[-1])
    with pytest.raises(ValueError, match='register must of shape'):
        _ = merge_qubits(regs, r1=qubits[:4], r2=qubits[5:7], r3=qubits[-1])
    merged_qregs = merge_qubits(regs, r1=qubits[:5], r2=qubits[5:7], r3=qubits[-1])
    assert merged_qregs == qubits
    expected_named_qubits = {'r1': cirq.NamedQubit.range(5, prefix='r1'), 'r2': cirq.NamedQubit.range(2, prefix='r2'), 'r3': [cirq.NamedQubit('r3')]}
    named_qregs = get_named_qubits(regs)
    for reg_name in expected_named_qubits:
        assert np.array_equal(named_qregs[reg_name], expected_named_qubits[reg_name])
    for reg_order in [[r1, r2, r3], [r2, r3, r1]]:
        flat_named_qubits = [q for v in get_named_qubits(cirq_ft.Signature(reg_order)).values() for q in v]
        expected_qubits = [q for r in reg_order for q in expected_named_qubits[r.name]]
        assert flat_named_qubits == expected_qubits