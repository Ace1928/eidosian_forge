from typing import Any, Dict, Iterable, List, Optional, Sequence, Union
from collections import defaultdict
import itertools
import random
import numpy as np
import sympy
from cirq import circuits, ops, linalg, protocols, qis
from cirq.testing import lin_alg_utils
def assert_circuits_with_terminal_measurements_are_equivalent(actual: circuits.AbstractCircuit, reference: circuits.AbstractCircuit, atol: float=1e-08) -> None:
    """Determines if two circuits have equivalent effects.

    The circuits can contain measurements, but the measurements must be at the
    end of the circuit. Circuits are equivalent if, for all possible inputs,
    their outputs (classical bits for lines terminated with measurement and
    qubits for lines without measurement) are observationally indistinguishable
    up to a tolerance. Note that under this definition of equivalence circuits
    that differ solely in the overall phase of the post-measurement state of
    measured qubits are considered equivalent.

    For example, applying an extra Z gate to an unmeasured qubit changes the
    effect of a circuit. But inserting a Z gate operation just before a
    measurement does not.

    Args:
        actual: The circuit that was actually computed by some process.
        reference: A circuit with the correct function.
        atol: Absolute error tolerance.
    """
    __tracebackhide__ = True
    measured_qubits_actual = {qubit for op in actual.all_operations() if protocols.is_measurement(op) for qubit in op.qubits}
    measured_qubits_reference = {qubit for op in reference.all_operations() if protocols.is_measurement(op) for qubit in op.qubits}
    assert actual.are_all_measurements_terminal()
    assert reference.are_all_measurements_terminal()
    assert measured_qubits_actual == measured_qubits_reference
    all_qubits = actual.all_qubits().union(reference.all_qubits())
    matrix_actual = actual.unitary(qubits_that_should_be_present=all_qubits)
    matrix_reference = reference.unitary(qubits_that_should_be_present=all_qubits)
    n_qubits = len(all_qubits)
    n = matrix_actual.shape[0]
    assert n == 1 << n_qubits
    assert matrix_actual.shape == matrix_reference.shape == (n, n)
    subspaces = _measurement_subspaces(measured_qubits_actual, n_qubits)
    for subspace in subspaces:
        block_actual = matrix_actual[subspace, :]
        block_reference = matrix_reference[subspace, :]
        assert linalg.allclose_up_to_global_phase(block_actual, block_reference, atol=atol), f"Circuit's effect differs from the reference circuit.\n\nDiagram of actual circuit:\n{actual}\n\nDiagram of reference circuit with desired function:\n{reference}\n"