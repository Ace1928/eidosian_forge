from typing import Optional
from torch import Tensor, tensor
from torchmetrics.functional.classification.auroc import binary_auroc
from torchmetrics.utilities.checks import _check_retrieval_functional_inputs
def retrieval_auroc(preds: Tensor, target: Tensor, top_k: Optional[int]=None, max_fpr: Optional[float]=None) -> Tensor:
    """Compute area under the receiver operating characteristic curve (AUROC) for information retrieval.

    ``preds`` and ``target`` should be of the same shape and live on the same device. If no ``target`` is ``True``,
    ``0`` is returned. ``target`` must be either `bool` or `integers` and ``preds`` must be ``float``,
    otherwise an error is raised.

    Args:
        preds: estimated probabilities of each document to be relevant.
        target: ground truth about each document being relevant or not.
        top_k: consider only the top k elements (default: ``None``, which considers them all)
        max_fpr: If not ``None``, calculates standardized partial AUC over the range ``[0, max_fpr]``.

    Return:
        a single-value tensor with the auroc value of the predictions ``preds`` w.r.t. the labels ``target``.

    Raises:
        ValueError:
            If ``top_k`` is not ``None`` or an integer larger than 0.

    Example:
        >>> from torchmetrics.functional.retrieval import retrieval_auroc
        >>> preds = tensor([0.2, 0.3, 0.5])
        >>> target = tensor([True, False, True])
        >>> retrieval_auroc(preds, target)
        tensor(0.5000)

    """
    preds, target = _check_retrieval_functional_inputs(preds, target)
    top_k = top_k or preds.shape[-1]
    if not (isinstance(top_k, int) and top_k > 0):
        raise ValueError('`top_k` has to be a positive integer or None')
    top_k_idx = preds.topk(min(top_k, preds.shape[-1]), sorted=True, dim=-1)[1]
    target = target[top_k_idx]
    if 0 not in target or 1 not in target:
        return tensor(0.0, device=preds.device, dtype=preds.dtype)
    preds = preds[top_k_idx]
    return binary_auroc(preds, target.int(), max_fpr=max_fpr)