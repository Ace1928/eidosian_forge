from typing import Optional, Union
import torch
from torch import Tensor, tensor
from typing_extensions import Literal
from torchmetrics.utilities.checks import _check_same_shape
def calculate_contingency_matrix(preds: Tensor, target: Tensor, eps: Optional[float]=None, sparse: bool=False) -> Tensor:
    """Calculate contingency matrix.

    Args:
        preds: predicted labels
        target: ground truth labels
        eps: value added to contingency matrix
        sparse: If True, returns contingency matrix as a sparse matrix. Else, return as dense matrix.
            `eps` must be `None` if `sparse` is `True`.

    Returns:
        contingency: contingency matrix of shape (n_classes_target, n_classes_preds)

    Example:
        >>> import torch
        >>> from torchmetrics.functional.clustering.utils import calculate_contingency_matrix
        >>> preds = torch.tensor([2, 1, 0, 1, 0])
        >>> target = torch.tensor([0, 2, 1, 1, 0])
        >>> calculate_contingency_matrix(preds, target, eps=1e-16)
        tensor([[1.0000e+00, 1.0000e-16, 1.0000e+00],
                [1.0000e+00, 1.0000e+00, 1.0000e-16],
                [1.0000e-16, 1.0000e+00, 1.0000e-16]])

    """
    if eps is not None and sparse is True:
        raise ValueError('Cannot specify `eps` and return sparse tensor.')
    if preds.ndim != 1 or target.ndim != 1:
        raise ValueError(f'Expected 1d `preds` and `target` but got {preds.ndim} and {target.dim}.')
    preds_classes, preds_idx = torch.unique(preds, return_inverse=True)
    target_classes, target_idx = torch.unique(target, return_inverse=True)
    num_classes_preds = preds_classes.size(0)
    num_classes_target = target_classes.size(0)
    contingency = torch.sparse_coo_tensor(torch.stack((target_idx, preds_idx)), torch.ones(target_idx.shape[0], dtype=preds_idx.dtype, device=preds_idx.device), (num_classes_target, num_classes_preds))
    if not sparse:
        contingency = contingency.to_dense()
        if eps:
            contingency = contingency + eps
    return contingency