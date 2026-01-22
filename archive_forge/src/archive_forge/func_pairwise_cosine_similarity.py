from typing import Optional
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.functional.pairwise.helpers import _check_input, _reduce_distance_matrix
from torchmetrics.utilities.compute import _safe_matmul
def pairwise_cosine_similarity(x: Tensor, y: Optional[Tensor]=None, reduction: Literal['mean', 'sum', 'none', None]=None, zero_diagonal: Optional[bool]=None) -> Tensor:
    """Calculate pairwise cosine similarity.

    .. math::
        s_{cos}(x,y) = \\frac{<x,y>}{||x|| \\cdot ||y||}
                     = \\frac{\\sum_{d=1}^D x_d \\cdot y_d }{\\sqrt{\\sum_{d=1}^D x_i^2} \\cdot \\sqrt{\\sum_{d=1}^D y_i^2}}

    If both :math:`x` and :math:`y` are passed in, the calculation will be performed pairwise
    between the rows of :math:`x` and :math:`y`.
    If only :math:`x` is passed in, the calculation will be performed between the rows of :math:`x`.

    Args:
        x: Tensor with shape ``[N, d]``
        y: Tensor with shape ``[M, d]``, optional
        reduction: reduction to apply along the last dimension. Choose between `'mean'`, `'sum'`
            (applied along column dimension) or  `'none'`, `None` for no reduction
        zero_diagonal: if the diagonal of the distance matrix should be set to 0. If only :math:`x` is given
            this defaults to ``True`` else if :math:`y` is also given it defaults to ``False``

    Returns:
        A ``[N,N]`` matrix of distances if only ``x`` is given, else a ``[N,M]`` matrix

    Example:
        >>> import torch
        >>> from torchmetrics.functional.pairwise import pairwise_cosine_similarity
        >>> x = torch.tensor([[2, 3], [3, 5], [5, 8]], dtype=torch.float32)
        >>> y = torch.tensor([[1, 0], [2, 1]], dtype=torch.float32)
        >>> pairwise_cosine_similarity(x, y)
        tensor([[0.5547, 0.8682],
                [0.5145, 0.8437],
                [0.5300, 0.8533]])
        >>> pairwise_cosine_similarity(x)
        tensor([[0.0000, 0.9989, 0.9996],
                [0.9989, 0.0000, 0.9998],
                [0.9996, 0.9998, 0.0000]])

    """
    distance = _pairwise_cosine_similarity_update(x, y, zero_diagonal)
    return _reduce_distance_matrix(distance, reduction)