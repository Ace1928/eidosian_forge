from typing import List, Optional, Tuple, Union
import torch
from torch import Tensor, tensor
from typing_extensions import Literal
from torchmetrics.utilities.checks import _check_same_shape, _input_format_classification
from torchmetrics.utilities.data import _bincount, select_topk
from torchmetrics.utilities.enums import AverageMethod, ClassificationTask, DataType, MDMCAverageMethod
def multiclass_stat_scores(preds: Tensor, target: Tensor, num_classes: int, average: Optional[Literal['micro', 'macro', 'weighted', 'none']]='macro', top_k: int=1, multidim_average: Literal['global', 'samplewise']='global', ignore_index: Optional[int]=None, validate_args: bool=True) -> Tensor:
    """Compute the true positives, false positives, true negatives, false negatives and support for multiclass tasks.

    Related to `Type I and Type II errors`_.

    Accepts the following input tensors:

    - ``preds``: ``(N, ...)`` (int tensor) or ``(N, C, ..)`` (float tensor). If preds is a floating point
      we apply ``torch.argmax`` along the ``C`` dimension to automatically convert probabilities/logits into
      an int tensor.
    - ``target`` (int tensor): ``(N, ...)``

    Args:
        preds: Tensor with predictions
        target: Tensor with true labels
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
        The metric returns a tensor of shape ``(..., 5)``, where the last dimension corresponds
        to ``[tp, fp, tn, fn, sup]`` (``sup`` stands for support and equals ``tp + fn``). The shape
        depends on ``average`` and ``multidim_average`` parameters:

        - If ``multidim_average`` is set to ``global``:

          - If ``average='micro'/'macro'/'weighted'``, the shape will be ``(5,)``
          - If ``average=None/'none'``, the shape will be ``(C, 5)``

        - If ``multidim_average`` is set to ``samplewise``:

          - If ``average='micro'/'macro'/'weighted'``, the shape will be ``(N, 5)``
          - If ``average=None/'none'``, the shape will be ``(N, C, 5)``

    Example (preds is int tensor):
        >>> from torch import tensor
        >>> from torchmetrics.functional.classification import multiclass_stat_scores
        >>> target = tensor([2, 1, 0, 0])
        >>> preds = tensor([2, 1, 0, 1])
        >>> multiclass_stat_scores(preds, target, num_classes=3, average='micro')
        tensor([3, 1, 7, 1, 4])
        >>> multiclass_stat_scores(preds, target, num_classes=3, average=None)
        tensor([[1, 0, 2, 1, 2],
                [1, 1, 2, 0, 1],
                [1, 0, 3, 0, 1]])

    Example (preds is float tensor):
        >>> from torchmetrics.functional.classification import multiclass_stat_scores
        >>> target = tensor([2, 1, 0, 0])
        >>> preds = tensor([[0.16, 0.26, 0.58],
        ...                 [0.22, 0.61, 0.17],
        ...                 [0.71, 0.09, 0.20],
        ...                 [0.05, 0.82, 0.13]])
        >>> multiclass_stat_scores(preds, target, num_classes=3, average='micro')
        tensor([3, 1, 7, 1, 4])
        >>> multiclass_stat_scores(preds, target, num_classes=3, average=None)
        tensor([[1, 0, 2, 1, 2],
                [1, 1, 2, 0, 1],
                [1, 0, 3, 0, 1]])

    Example (multidim tensors):
        >>> from torchmetrics.functional.classification import multiclass_stat_scores
        >>> target = tensor([[[0, 1], [2, 1], [0, 2]], [[1, 1], [2, 0], [1, 2]]])
        >>> preds = tensor([[[0, 2], [2, 0], [0, 1]], [[2, 2], [2, 1], [1, 0]]])
        >>> multiclass_stat_scores(preds, target, num_classes=3, multidim_average='samplewise', average='micro')
        tensor([[3, 3, 9, 3, 6],
                [2, 4, 8, 4, 6]])
        >>> multiclass_stat_scores(preds, target, num_classes=3, multidim_average='samplewise', average=None)
        tensor([[[2, 1, 3, 0, 2],
                 [0, 1, 3, 2, 2],
                 [1, 1, 3, 1, 2]],
                [[0, 1, 4, 1, 1],
                 [1, 1, 2, 2, 3],
                 [1, 2, 2, 1, 2]]])

    """
    if validate_args:
        _multiclass_stat_scores_arg_validation(num_classes, top_k, average, multidim_average, ignore_index)
        _multiclass_stat_scores_tensor_validation(preds, target, num_classes, multidim_average, ignore_index)
    preds, target = _multiclass_stat_scores_format(preds, target, top_k)
    tp, fp, tn, fn = _multiclass_stat_scores_update(preds, target, num_classes, top_k, average, multidim_average, ignore_index)
    return _multiclass_stat_scores_compute(tp, fp, tn, fn, average, multidim_average)