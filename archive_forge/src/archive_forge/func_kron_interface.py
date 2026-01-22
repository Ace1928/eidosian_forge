import itertools
from functools import reduce
from typing import Generator, Iterable, Tuple
import numpy as np
from scipy.sparse import csr_matrix, eye, kron
import pennylane as qml
from pennylane.wires import Wires
def kron_interface(mat1, mat2):
    if interface == 'scipy':
        res = kron(mat1, mat2, format='coo')
        res.eliminate_zeros()
        return res
    if interface == 'torch':
        mat1 = mat1.contiguous()
        mat2 = mat2.contiguous()
    return qml.math.kron(mat1, mat2, like=interface)