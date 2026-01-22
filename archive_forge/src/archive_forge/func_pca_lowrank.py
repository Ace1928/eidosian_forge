from typing import Optional, Tuple
import torch
from torch import Tensor
from . import _linalg_utils as _utils
from .overrides import handle_torch_function, has_torch_function
def pca_lowrank(A: Tensor, q: Optional[int]=None, center: bool=True, niter: int=2) -> Tuple[Tensor, Tensor, Tensor]:
    """Performs linear Principal Component Analysis (PCA) on a low-rank
    matrix, batches of such matrices, or sparse matrix.

    This function returns a namedtuple ``(U, S, V)`` which is the
    nearly optimal approximation of a singular value decomposition of
    a centered matrix :math:`A` such that :math:`A = U diag(S) V^T`.

    .. note:: The relation of ``(U, S, V)`` to PCA is as follows:

                - :math:`A` is a data matrix with ``m`` samples and
                  ``n`` features

                - the :math:`V` columns represent the principal directions

                - :math:`S ** 2 / (m - 1)` contains the eigenvalues of
                  :math:`A^T A / (m - 1)` which is the covariance of
                  ``A`` when ``center=True`` is provided.

                - ``matmul(A, V[:, :k])`` projects data to the first k
                  principal components

    .. note:: Different from the standard SVD, the size of returned
              matrices depend on the specified rank and q
              values as follows:

                - :math:`U` is m x q matrix

                - :math:`S` is q-vector

                - :math:`V` is n x q matrix

    .. note:: To obtain repeatable results, reset the seed for the
              pseudorandom number generator

    Args:

        A (Tensor): the input tensor of size :math:`(*, m, n)`

        q (int, optional): a slightly overestimated rank of
                           :math:`A`. By default, ``q = min(6, m,
                           n)``.

        center (bool, optional): if True, center the input tensor,
                                 otherwise, assume that the input is
                                 centered.

        niter (int, optional): the number of subspace iterations to
                               conduct; niter must be a nonnegative
                               integer, and defaults to 2.

    References::

        - Nathan Halko, Per-Gunnar Martinsson, and Joel Tropp, Finding
          structure with randomness: probabilistic algorithms for
          constructing approximate matrix decompositions,
          arXiv:0909.4061 [math.NA; math.PR], 2009 (available at
          `arXiv <http://arxiv.org/abs/0909.4061>`_).

    """
    if not torch.jit.is_scripting():
        if type(A) is not torch.Tensor and has_torch_function((A,)):
            return handle_torch_function(pca_lowrank, (A,), A, q=q, center=center, niter=niter)
    m, n = A.shape[-2:]
    if q is None:
        q = min(6, m, n)
    elif not (q >= 0 and q <= min(m, n)):
        raise ValueError(f'q(={q}) must be non-negative integer and not greater than min(m, n)={min(m, n)}')
    if not niter >= 0:
        raise ValueError(f'niter(={niter}) must be non-negative integer')
    dtype = _utils.get_floating_dtype(A)
    if not center:
        return _svd_lowrank(A, q, niter=niter, M=None)
    if _utils.is_sparse(A):
        if len(A.shape) != 2:
            raise ValueError('pca_lowrank input is expected to be 2-dimensional tensor')
        c = torch.sparse.sum(A, dim=(-2,)) / m
        column_indices = c.indices()[0]
        indices = torch.zeros(2, len(column_indices), dtype=column_indices.dtype, device=column_indices.device)
        indices[0] = column_indices
        C_t = torch.sparse_coo_tensor(indices, c.values(), (n, 1), dtype=dtype, device=A.device)
        ones_m1_t = torch.ones(A.shape[:-2] + (1, m), dtype=dtype, device=A.device)
        M = _utils.transpose(torch.sparse.mm(C_t, ones_m1_t))
        return _svd_lowrank(A, q, niter=niter, M=M)
    else:
        C = A.mean(dim=(-2,), keepdim=True)
        return _svd_lowrank(A - C, q, niter=niter, M=None)