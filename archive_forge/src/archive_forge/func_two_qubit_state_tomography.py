import dataclasses
import itertools
from typing import Any, cast, Iterator, List, Optional, Sequence, Tuple, TYPE_CHECKING
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from cirq import circuits, ops, protocols
def two_qubit_state_tomography(sampler: 'cirq.Sampler', first_qubit: 'cirq.Qid', second_qubit: 'cirq.Qid', circuit: 'cirq.AbstractCircuit', repetitions: int=1000) -> TomographyResult:
    """Two-qubit state tomography.

    To measure the density matrix of the output state of a two-qubit circuit,
    different combinations of I, X/2 and Y/2 operations are applied to the
    two qubits before measurements in the z-basis to determine the state
    probabilities $P_{00}, P_{01}, P_{10}.$

    The density matrix rho is decomposed into an operator-sum representation
    $\\sum_{i, j} c_{ij} * \\sigma_i \\bigotimes \\sigma_j$, where $i, j = 0, 1, 2,
    3$ and $\\sigma_0 = I, \\sigma_1 = \\sigma_x, \\sigma_2 = \\sigma_y, \\sigma_3 =
    \\sigma_z$ are the single-qubit Identity and Pauli matrices.

    Based on the measured probabilities probs and the transformations of the
    measurement operator by different basis rotations, one can build an
    overdetermined set of linear equations.

    As an example, if the identity operation (I) is applied to both qubits, the
    measurement operators are $(I +/- \\sigma_z) \\bigotimes (I +/- \\sigma_z)$.
    The state probabilities $P_{00}, P_{01}, P_{10}$ thus obtained contribute
    to the following linear equations (setting $c_{00} = 1$):

    $$
    \\begin{align}
    c_{03} + c_{30} + c_{33} &= 4*P_{00} - 1 \\\\
    -c_{03} + c_{30} - c_{33} &= 4*P_{01} - 1 \\\\
    c_{03} - c_{30} - c_{33} &= 4*P_{10} - 1
    \\end{align}
    $$

    And if a Y/2 rotation is applied to the first qubit and a X/2 rotation
    is applied to the second qubit before measurement, the measurement
    operators are $(I -/+ \\sigma_x) \\bigotimes (I +/- \\sigma_y)$. The
    probabilities obtained instead contribute to the following linear equations:

    $$
    \\begin{align}
    c_{02} - c_{10} - c_{12} &= 4*P_{00} - 1 \\\\
    -c_{02} - c_{10} + c_{12} &= 4*P_{01} - 1 \\\\
    c_{02} + c_{10} + c_{12} &= 4*P_{10} - 1
    \\end{align}
    $$

    Note that this set of equations has the same form as the first set under
    the transformation $c_{03}$ <-> $c_{02}, c_{30}$ <-> $-c_{10}$ and
    $c_{33}$ <-> $-c_{12}$.

    Since there are 9 possible combinations of rotations (each producing 3
    independent probabilities) and a total of 15 unknown coefficients $c_{ij}$,
    one can cast all the measurement results into a overdetermined set of
    linear equations numpy.dot(mat, c) = probs. Here c is of length 15 and
    contains all the $c_{ij}$'s (except $c_{00}$ which is set to 1), and mat
    is a 27 by 15 matrix having three non-zero elements in each row that are
    either 1 or -1.

    The least-square solution to the above set of linear equations is then
    used to construct the density matrix rho.

    See Vandersypen and Chuang, Rev. Mod. Phys. 76, 1037 for details and
    Steffen et al, Science 313, 1423 for a related experiment.

    Args:
        sampler: The quantum engine or simulator to run the circuits.
        first_qubit: The first qubit under test.
        second_qubit: The second qubit under test.
        circuit: The circuit to execute on the qubits before tomography.
        repetitions: The number of measurements for each basis rotation.

    Returns:
        A TomographyResult object that stores and plots the density matrix.
    """
    num_rows = 27
    num_cols = 15

    def _measurement(two_qubit_circuit: circuits.Circuit) -> np.ndarray:
        two_qubit_circuit.append(ops.measure(first_qubit, second_qubit, key='z'))
        results = sampler.run(two_qubit_circuit, repetitions=repetitions)
        results_hist = results.histogram(key='z')
        prob_list = [results_hist[0], results_hist[1], results_hist[2]]
        return np.asarray(prob_list) / repetitions
    sigma_0 = np.eye(2) * 0.5
    sigma_1 = np.array([[0.0, 1.0], [1.0, 0.0]]) * 0.5
    sigma_2 = np.array([[0.0, -1j], [1j, 0.0]]) * 0.5
    sigma_3 = np.array([[1.0, 0.0], [0.0, -1.0]]) * 0.5
    sigmas = [sigma_0, sigma_1, sigma_2, sigma_3]
    probs: np.ndarray = np.array([])
    rots = [ops.X ** 0, ops.X ** 0.5, ops.Y ** 0.5]
    mat = np.zeros((num_rows, num_cols))
    s = np.array([[1.0, 1.0, 1.0], [-1.0, 1.0, -1.0], [1.0, -1.0, -1.0]])
    for i, rot_1 in enumerate(rots):
        for j, rot_2 in enumerate(rots):
            m_idx, indices, signs = _indices_after_basis_rot(i, j)
            mat[m_idx:m_idx + 3, indices] = s * np.tile(signs, (3, 1))
            test_circuit = circuit + circuits.Circuit(rot_1(first_qubit))
            test_circuit.append(rot_2(second_qubit))
            probs = np.concatenate((probs, _measurement(test_circuit)))
    c, _, _, _ = np.linalg.lstsq(mat, 4.0 * probs - 1.0, rcond=-1)
    c = np.concatenate(([1.0], c))
    c = c.reshape(4, 4)
    rho = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            rho = rho + c[i, j] * np.kron(sigmas[i], sigmas[j])
    return TomographyResult(rho)