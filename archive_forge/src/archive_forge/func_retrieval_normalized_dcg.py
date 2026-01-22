from typing import Optional
import torch
from torch import Tensor
from torchmetrics.utilities.checks import _check_retrieval_functional_inputs
def retrieval_normalized_dcg(preds: Tensor, target: Tensor, top_k: Optional[int]=None) -> Tensor:
    """Compute `Normalized Discounted Cumulative Gain`_ (for information retrieval).

    ``preds`` and ``target`` should be of the same shape and live on the same device.
    ``target`` must be either `bool` or `integers` and ``preds`` must be ``float``,
    otherwise an error is raised.

    Args:
        preds: estimated probabilities of each document to be relevant.
        target: ground truth about each document relevance.
        top_k: consider only the top k elements (default: ``None``, which considers them all)

    Return:
        A single-value tensor with the nDCG of the predictions ``preds`` w.r.t. the labels ``target``.

    Raises:
        ValueError:
            If ``top_k`` parameter is not `None` or an integer larger than 0

    Example:
        >>> from torchmetrics.functional.retrieval import retrieval_normalized_dcg
        >>> preds = torch.tensor([.1, .2, .3, 4, 70])
        >>> target = torch.tensor([10, 0, 0, 1, 5])
        >>> retrieval_normalized_dcg(preds, target)
        tensor(0.6957)

    """
    preds, target = _check_retrieval_functional_inputs(preds, target, allow_non_binary_target=True)
    top_k = preds.shape[-1] if top_k is None else top_k
    if not (isinstance(top_k, int) and top_k > 0):
        raise ValueError('`top_k` has to be a positive integer or None')
    gain = _dcg_sample_scores(target, preds, top_k, ignore_ties=False)
    normalized_gain = _dcg_sample_scores(target, target, top_k, ignore_ties=True)
    all_irrelevant = normalized_gain == 0
    gain[all_irrelevant] = 0
    gain[~all_irrelevant] /= normalized_gain[~all_irrelevant]
    return gain.mean()