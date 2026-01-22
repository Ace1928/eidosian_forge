from typing import Optional
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.functional.pairwise.helpers import _check_input, _reduce_distance_matrix
def pairwise_manhattan_distance(x: Tensor, y: Optional[Tensor]=None, reduction: Literal['mean', 'sum', 'none', None]=None, zero_diagonal: Optional[bool]=None) -> Tensor:
    """Calculate pairwise manhattan distance.

    .. math::
        d_{man}(x,y) = ||x-y||_1 = \\sum_{d=1}^D |x_d - y_d|

    If both :math:`x` and :math:`y` are passed in, the calculation will be performed pairwise between
    the rows of :math:`x` and :math:`y`.
    If only :math:`x` is passed in, the calculation will be performed between the rows of :math:`x`.

    Args:
        x: Tensor with shape ``[N, d]``
        y: Tensor with shape ``[M, d]``, optional
        reduction: reduction to apply along the last dimension. Choose between `'mean'`, `'sum'`
            (applied along column dimension) or  `'none'`, `None` for no reduction
        zero_diagonal: if the diagonal of the distance matrix should be set to 0. If only `x` is given
            this defaults to `True` else if `y` is also given it defaults to `False`

    Returns:
        A ``[N,N]`` matrix of distances if only ``x`` is given, else a ``[N,M]`` matrix

    Example:
        >>> import torch
        >>> from torchmetrics.functional.pairwise import pairwise_manhattan_distance
        >>> x = torch.tensor([[2, 3], [3, 5], [5, 8]], dtype=torch.float32)
        >>> y = torch.tensor([[1, 0], [2, 1]], dtype=torch.float32)
        >>> pairwise_manhattan_distance(x, y)
        tensor([[ 4.,  2.],
                [ 7.,  5.],
                [12., 10.]])
        >>> pairwise_manhattan_distance(x)
        tensor([[0., 3., 8.],
                [3., 0., 5.],
                [8., 5., 0.]])

    """
    distance = _pairwise_manhattan_distance_update(x, y, zero_diagonal)
    return _reduce_distance_matrix(distance, reduction)