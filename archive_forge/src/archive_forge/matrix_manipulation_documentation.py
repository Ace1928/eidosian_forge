import itertools
from functools import reduce
from typing import Generator, Iterable, Tuple
import numpy as np
from scipy.sparse import csr_matrix, eye, kron
import pennylane as qml
from pennylane.wires import Wires

    Determine whether a tensor has an additional batch dimension for broadcasting,
    compared to an expected_shape. Has support for abstract TF tensors.

    Args:
        tensor (TensorLike): A tensor to inspect for batching
        expected_shape (Tuple[int]): The expected shape of the tensor if not batched
        expected_size (int): The expected size of the tensor if not batched

    Returns:
        Optional[int]: The batch size of the tensor if there is one, otherwise None
    