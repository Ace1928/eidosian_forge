from __future__ import annotations
import collections
import itertools
import math
import operator
import warnings
from fractions import Fraction
from functools import reduce
from typing import TYPE_CHECKING, cast
import numpy as np
from monty.dev import deprecated
from monty.json import MSONable
from scipy.spatial import Voronoi
from pymatgen.util.coord import pbc_shortest_vectors
from pymatgen.util.due import Doi, due
@due.dcite(Doi('10.1107/S010876730302186X'), description='Numerically stable algorithms for the computation of reduced unit cells')
def get_niggli_reduced_lattice(self, tol: float=1e-05) -> Lattice:
    """Get the Niggli reduced lattice using the numerically stable algo
        proposed by R. W. Grosse-Kunstleve, N. K. Sauter, & P. D. Adams,
        Acta Crystallographica Section A Foundations of Crystallography, 2003,
        60(1), 1-6. doi:10.1107/S010876730302186X.

        Args:
            tol (float): The numerical tolerance. The default of 1e-5 should
                result in stable behavior for most cases.

        Returns:
            Lattice: Niggli-reduced lattice.
        """
    matrix = self.lll_matrix
    e = tol * self.volume ** (1 / 3)
    G = np.dot(matrix, matrix.T)
    for _ in range(100):
        A, B, C, E, N, Y = (G[0, 0], G[1, 1], G[2, 2], 2 * G[1, 2], 2 * G[0, 2], 2 * G[0, 1])
        if B + e < A or (abs(A - B) < e and abs(E) > abs(N) + e):
            M = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]])
            G = np.dot(np.transpose(M), np.dot(G, M))
            A, B, C, E, N, Y = (G[0, 0], G[1, 1], G[2, 2], 2 * G[1, 2], 2 * G[0, 2], 2 * G[0, 1])
        if C + e < B or (abs(B - C) < e and abs(N) > abs(Y) + e):
            M = np.array([[-1, 0, 0], [0, 0, -1], [0, -1, 0]])
            G = np.dot(np.transpose(M), np.dot(G, M))
            continue
        ll = 0 if abs(E) < e else E / abs(E)
        m = 0 if abs(N) < e else N / abs(N)
        n = 0 if abs(Y) < e else Y / abs(Y)
        if ll * m * n == 1:
            i = -1 if ll == -1 else 1
            j = -1 if m == -1 else 1
            k = -1 if n == -1 else 1
            M = np.diag((i, j, k))
            G = np.dot(np.transpose(M), np.dot(G, M))
        elif ll * m * n in (0, -1):
            i = -1 if ll == 1 else 1
            j = -1 if m == 1 else 1
            k = -1 if n == 1 else 1
            if i * j * k == -1:
                if n == 0:
                    k = -1
                elif m == 0:
                    j = -1
                elif ll == 0:
                    i = -1
            M = np.diag((i, j, k))
            G = np.dot(np.transpose(M), np.dot(G, M))
        A, B, C, E, N, Y = (G[0, 0], G[1, 1], G[2, 2], 2 * G[1, 2], 2 * G[0, 2], 2 * G[0, 1])
        if abs(E) > B + e or (abs(E - B) < e and Y - e > 2 * N) or (abs(E + B) < e and -e > Y):
            M = np.array([[1, 0, 0], [0, 1, -E / abs(E)], [0, 0, 1]])
            G = np.dot(np.transpose(M), np.dot(G, M))
            continue
        if abs(N) > A + e or (abs(A - N) < e and Y - e > 2 * E) or (abs(A + N) < e and -e > Y):
            M = np.array([[1, 0, -N / abs(N)], [0, 1, 0], [0, 0, 1]])
            G = np.dot(np.transpose(M), np.dot(G, M))
            continue
        if abs(Y) > A + e or (abs(A - Y) < e and N - e > 2 * E) or (abs(A + Y) < e and -e > N):
            M = np.array([[1, -Y / abs(Y), 0], [0, 1, 0], [0, 0, 1]])
            G = np.dot(np.transpose(M), np.dot(G, M))
            continue
        if -e > E + N + Y + A + B or abs(E + N + Y + A + B) < e < Y + (A + N) * 2:
            M = np.array([[1, 0, 1], [0, 1, 1], [0, 0, 1]])
            G = np.dot(np.transpose(M), np.dot(G, M))
            continue
        break
    A = G[0, 0]
    B = G[1, 1]
    C = G[2, 2]
    E = 2 * G[1, 2]
    N = 2 * G[0, 2]
    Y = 2 * G[0, 1]
    a = math.sqrt(A)
    b = math.sqrt(B)
    c = math.sqrt(C)
    alpha = math.acos(E / 2 / b / c) / math.pi * 180
    beta = math.acos(N / 2 / a / c) / math.pi * 180
    gamma = math.acos(Y / 2 / a / b) / math.pi * 180
    lattice = Lattice.from_parameters(a, b, c, alpha, beta, gamma)
    mapped = self.find_mapping(lattice, e, skip_rotation_matrix=True)
    if mapped is not None:
        if np.linalg.det(mapped[0].matrix) > 0:
            return mapped[0]
        return Lattice(-mapped[0].matrix)
    raise ValueError("can't find niggli")