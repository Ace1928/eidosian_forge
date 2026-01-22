from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable
import numpy as np
import scipy.sparse as sp
from scipy.signal import convolve
from cvxpy.lin_ops import LinOp
from cvxpy.settings import (
def get_tensor_representation(self, row_offset: int, total_rows: int) -> TensorRepresentation:
    """
        Returns a TensorRepresentation of [A b] tensor.
        This function iterates through all the tensor data and constructs their
        respective representation in COO format. The row data is adjusted according
        to the position of each element within a parameter slice. The parameter_offset
        finds which slice the original row indices belong to before applying the column
        offset.
        """
    assert self.tensor is not None
    shape = (total_rows, self.var_length + 1)
    tensor_representations = []
    for variable_id, variable_tensor in self.tensor.items():
        for parameter_id, parameter_matrix in variable_tensor.items():
            p = self.param_to_size[parameter_id]
            m = parameter_matrix.shape[0] // p
            coo_repr = parameter_matrix.tocoo(copy=False)
            tensor_representations.append(TensorRepresentation(coo_repr.data, coo_repr.row % m + row_offset, coo_repr.col + self.id_to_col[variable_id], coo_repr.row // m + np.ones(coo_repr.nnz) * self.param_to_col[parameter_id], shape=shape))
    return TensorRepresentation.combine(tensor_representations)