import itertools
import cirq
import cirq_ft
import numpy as np
import pytest
from cirq_ft.infra import bit_tools
from cirq_ft.deprecation import allow_deprecated_cirq_ft_use_in_tests
@pytest.mark.parametrize('bitsize', [3])
@pytest.mark.parametrize('mod', [5, 8])
@pytest.mark.parametrize('add_val', [1, 2])
@pytest.mark.parametrize('cv', [[], [0, 1], [1, 0], [1, 1]])
@allow_deprecated_cirq_ft_use_in_tests
def test_add_mod_n(bitsize, mod, add_val, cv):
    gate = cirq_ft.AddMod(bitsize, mod, add_val=add_val, cv=cv)
    basis_map = {}
    num_cvs = len(cv)
    for x in range(2 ** bitsize):
        y = (x + add_val) % mod if x < mod else x
        if not num_cvs:
            basis_map[x] = y
            continue
        for cb in range(2 ** num_cvs):
            inp = f'0b_{cb:0{num_cvs}b}_{x:0{bitsize}b}'
            if tuple((int(x) for x in f'{cb:0{num_cvs}b}')) == tuple(cv):
                out = f'0b_{cb:0{num_cvs}b}_{y:0{bitsize}b}'
                basis_map[int(inp, 2)] = int(out, 2)
            else:
                basis_map[int(inp, 2)] = int(inp, 2)
    num_qubits = gate.num_qubits()
    op = gate.on(*cirq.LineQubit.range(num_qubits))
    circuit = cirq.Circuit(op)
    cirq.testing.assert_equivalent_computational_basis_map(basis_map, circuit)
    circuit += op ** (-1)
    cirq.testing.assert_equivalent_computational_basis_map(identity_map(num_qubits), circuit)
    cirq.testing.assert_equivalent_repr(gate, setup_code='import cirq_ft')