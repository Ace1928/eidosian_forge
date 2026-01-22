from typing import Optional, Tuple
import torch
from torch import Tensor
def transjugate(A):
    """Return transpose conjugate of a matrix or batches of matrices."""
    return conjugate(transpose(A))