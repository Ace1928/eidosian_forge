from __future__ import annotations
import typing
from functools import partial
from collections.abc import Callable
from typing import Protocol
import numpy as np
from qiskit.quantum_info import Operator
from .approximate import ApproximateCircuit, ApproximatingObjective
def compile_unitary(self, target_matrix: np.ndarray, approximate_circuit: ApproximateCircuit, approximating_objective: ApproximatingObjective, initial_point: np.ndarray | None=None) -> None:
    """
        Approximately compiles a circuit represented as a unitary matrix by solving an optimization
        problem defined by ``approximating_objective`` and using ``approximate_circuit`` as a
        template for the approximate circuit.

        Args:
            target_matrix: a unitary matrix to approximate.
            approximate_circuit: a template circuit that will be filled with the parameter values
                obtained in the optimization procedure.
            approximating_objective: a definition of the optimization problem.
            initial_point: initial values of angles/parameters to start optimization from.
        """
    matrix_dim = target_matrix.shape[0]
    target_det = np.linalg.det(target_matrix)
    if not np.isclose(target_det, 1):
        su_matrix = target_matrix / np.power(target_det, 1 / matrix_dim, dtype=complex)
        global_phase_required = True
    else:
        su_matrix = target_matrix
        global_phase_required = False
    approximating_objective.target_matrix = su_matrix
    if initial_point is None:
        np.random.seed(self._seed)
        initial_point = np.random.uniform(0, 2 * np.pi, approximating_objective.num_thetas)
    opt_result = self._optimizer(fun=approximating_objective.objective, x0=initial_point, jac=approximating_objective.gradient)
    approximate_circuit.build(opt_result.x)
    approx_matrix = Operator(approximate_circuit).data
    if global_phase_required:
        alpha = np.angle(np.trace(np.dot(approx_matrix.conj().T, target_matrix)))
        approximate_circuit.global_phase = alpha