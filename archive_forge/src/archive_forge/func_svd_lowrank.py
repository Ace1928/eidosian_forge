from typing import Optional, Tuple
import torch
from torch import Tensor
from . import _linalg_utils as _utils
from .overrides import handle_torch_function, has_torch_function
def svd_lowrank(A: Tensor, q: Optional[int]=6, niter: Optional[int]=2, M: Optional[Tensor]=None) -> Tuple[Tensor, Tensor, Tensor]:
    """Return the singular value decomposition ``(U, S, V)`` of a matrix,
    batches of matrices, or a sparse matrix :math:`A` such that
    :math:`A \\approx U diag(S) V^T`. In case :math:`M` is given, then
    SVD is computed for the matrix :math:`A - M`.

    .. note:: The implementation is based on the Algorithm 5.1 from
              Halko et al, 2009.

    .. note:: To obtain repeatable results, reset the seed for the
              pseudorandom number generator

    .. note:: The input is assumed to be a low-rank matrix.

    .. note:: In general, use the full-rank SVD implementation
              :func:`torch.linalg.svd` for dense matrices due to its 10-fold
              higher performance characteristics. The low-rank SVD
              will be useful for huge sparse matrices that
              :func:`torch.linalg.svd` cannot handle.

    Args::
        A (Tensor): the input tensor of size :math:`(*, m, n)`

        q (int, optional): a slightly overestimated rank of A.

        niter (int, optional): the number of subspace iterations to
                               conduct; niter must be a nonnegative
                               integer, and defaults to 2

        M (Tensor, optional): the input tensor's mean of size
                              :math:`(*, 1, n)`.

    References::
        - Nathan Halko, Per-Gunnar Martinsson, and Joel Tropp, Finding
          structure with randomness: probabilistic algorithms for
          constructing approximate matrix decompositions,
          arXiv:0909.4061 [math.NA; math.PR], 2009 (available at
          `arXiv <https://arxiv.org/abs/0909.4061>`_).

    """
    if not torch.jit.is_scripting():
        tensor_ops = (A, M)
        if not set(map(type, tensor_ops)).issubset((torch.Tensor, type(None))) and has_torch_function(tensor_ops):
            return handle_torch_function(svd_lowrank, tensor_ops, A, q=q, niter=niter, M=M)
    return _svd_lowrank(A, q=q, niter=niter, M=M)