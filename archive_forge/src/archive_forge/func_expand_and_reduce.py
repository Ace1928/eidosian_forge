import itertools
from functools import reduce
from typing import Generator, Iterable, Tuple
import numpy as np
from scipy.sparse import csr_matrix, eye, kron
import pennylane as qml
from pennylane.wires import Wires
def expand_and_reduce(op1_tuple: Tuple[np.ndarray, Wires], op2_tuple: Tuple[np.ndarray, Wires]):
    mat1, wires1 = op1_tuple
    mat2, wires2 = op2_tuple
    expanded_wires = wires1 + wires2
    mat1 = expand_matrix(mat1, wires1, wire_order=expanded_wires)
    mat2 = expand_matrix(mat2, wires2, wire_order=expanded_wires)
    return (reduce_func(mat1, mat2), expanded_wires)