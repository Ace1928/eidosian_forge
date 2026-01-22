from ._internal import NDArrayBase
from ..base import _Null
Straight-through-estimator of `sign()`.

    In forward pass, returns element-wise sign of the input (same as `sign()`).

    In backward pass, returns gradients of ``1`` everywhere (instead of ``0`` everywhere as in ``sign()``):
    :math:`\frac{d}{dx}{sign\_ste(x)} = 1` vs. :math:`\frac{d}{dx}{sign(x)} = 0`.
    This is useful for quantized training.

    Reference: Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation.

    Example::
      x = sign_ste([-2, 0, 3])
      x.backward()
      x = [-1.,  0., 1.]
      x.grad() = [1.,  1., 1.]

    The storage type of ``sign_ste`` output depends upon the input storage type:
      - round_ste(default) = default
      - round_ste(row_sparse) = row_sparse
      - round_ste(csr) = csr


    Defined in ../src/operator/contrib/stes_op.cc:L79

    Parameters
    ----------
    data : NDArray
        The input array.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    