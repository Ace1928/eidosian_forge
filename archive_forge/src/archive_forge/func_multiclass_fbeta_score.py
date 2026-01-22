from typing import Optional
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.functional.classification.stat_scores import (
from torchmetrics.utilities.compute import _adjust_weights_safe_divide, _safe_divide
from torchmetrics.utilities.enums import ClassificationTask
def multiclass_fbeta_score(preds: Tensor, target: Tensor, beta: float, num_classes: int, average: Optional[Literal['micro', 'macro', 'weighted', 'none']]='macro', top_k: int=1, multidim_average: Literal['global', 'samplewise']='global', ignore_index: Optional[int]=None, validate_args: bool=True) -> Tensor:
    """Compute `F-score`_ metric for multiclass tasks.

    .. math::
        F_{\\beta} = (1 + \\beta^2) * \\frac{\\text{precision} * \\text{recall}}
        {(\\beta^2 * \\text{precision}) + \\text{recall}}

    Accepts the following input tensors:

    - ``preds``: ``(N, ...)`` (int tensor) or ``(N, C, ..)`` (float tensor). If preds is a floating point
      we apply ``torch.argmax`` along the ``C`` dimension to automatically convert probabilities/logits into
      an int tensor.
    - ``target`` (int tensor): ``(N, ...)``

    Args:
        preds: Tensor with predictions
        target: Tensor with true labels
        beta: Weighting between precision and recall in calculation. Setting to 1 corresponds to equal weight
        num_classes: Integer specifying the number of classes
        average:
            Defines the reduction that is applied over labels. Should be one of the following:

            - ``micro``: Sum statistics over all labels
            - ``macro``: Calculate statistics for each label and average them
            - ``weighted``: calculates statistics for each label and computes weighted average using their support
            - ``"none"`` or ``None``: calculates statistic for each label and applies no reduction
        top_k:
            Number of highest probability or logit score predictions considered to find the correct label.
            Only works when ``preds`` contain probabilities/logits.
        multidim_average:
            Defines how additionally dimensions ``...`` should be handled. Should be one of the following:

            - ``global``: Additional dimensions are flatted along the batch dimension
            - ``samplewise``: Statistic will be calculated independently for each sample on the ``N`` axis.
              The statistics in this case are calculated over the additional dimensions.

        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.

    Returns:
        The returned shape depends on the ``average`` and ``multidim_average`` arguments:

        - If ``multidim_average`` is set to ``global``:

          - If ``average='micro'/'macro'/'weighted'``, the output will be a scalar tensor
          - If ``average=None/'none'``, the shape will be ``(C,)``

        - If ``multidim_average`` is set to ``samplewise``:

          - If ``average='micro'/'macro'/'weighted'``, the shape will be ``(N,)``
          - If ``average=None/'none'``, the shape will be ``(N, C)``

    Example (preds is int tensor):
        >>> from torch import tensor
        >>> from torchmetrics.functional.classification import multiclass_fbeta_score
        >>> target = tensor([2, 1, 0, 0])
        >>> preds = tensor([2, 1, 0, 1])
        >>> multiclass_fbeta_score(preds, target, beta=2.0, num_classes=3)
        tensor(0.7963)
        >>> multiclass_fbeta_score(preds, target, beta=2.0, num_classes=3, average=None)
        tensor([0.5556, 0.8333, 1.0000])

    Example (preds is float tensor):
        >>> from torchmetrics.functional.classification import multiclass_fbeta_score
        >>> target = tensor([2, 1, 0, 0])
        >>> preds = tensor([[0.16, 0.26, 0.58],
        ...                 [0.22, 0.61, 0.17],
        ...                 [0.71, 0.09, 0.20],
        ...                 [0.05, 0.82, 0.13]])
        >>> multiclass_fbeta_score(preds, target, beta=2.0, num_classes=3)
        tensor(0.7963)
        >>> multiclass_fbeta_score(preds, target, beta=2.0, num_classes=3, average=None)
        tensor([0.5556, 0.8333, 1.0000])

    Example (multidim tensors):
        >>> from torchmetrics.functional.classification import multiclass_fbeta_score
        >>> target = tensor([[[0, 1], [2, 1], [0, 2]], [[1, 1], [2, 0], [1, 2]]])
        >>> preds = tensor([[[0, 2], [2, 0], [0, 1]], [[2, 2], [2, 1], [1, 0]]])
        >>> multiclass_fbeta_score(preds, target, beta=2.0, num_classes=3, multidim_average='samplewise')
        tensor([0.4697, 0.2706])
        >>> multiclass_fbeta_score(preds, target, beta=2.0, num_classes=3, multidim_average='samplewise', average=None)
        tensor([[0.9091, 0.0000, 0.5000],
                [0.0000, 0.3571, 0.4545]])

    """
    if validate_args:
        _multiclass_fbeta_score_arg_validation(beta, num_classes, top_k, average, multidim_average, ignore_index)
        _multiclass_stat_scores_tensor_validation(preds, target, num_classes, multidim_average, ignore_index)
    preds, target = _multiclass_stat_scores_format(preds, target, top_k)
    tp, fp, tn, fn = _multiclass_stat_scores_update(preds, target, num_classes, top_k, average, multidim_average, ignore_index)
    return _fbeta_reduce(tp, fp, tn, fn, beta, average=average, multidim_average=multidim_average)