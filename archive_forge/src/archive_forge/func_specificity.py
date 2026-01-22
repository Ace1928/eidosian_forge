from typing import Optional
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.functional.classification.stat_scores import (
from torchmetrics.utilities.compute import _adjust_weights_safe_divide, _safe_divide
from torchmetrics.utilities.enums import ClassificationTask
def specificity(preds: Tensor, target: Tensor, task: Literal['binary', 'multiclass', 'multilabel'], threshold: float=0.5, num_classes: Optional[int]=None, num_labels: Optional[int]=None, average: Optional[Literal['micro', 'macro', 'weighted', 'none']]='micro', multidim_average: Optional[Literal['global', 'samplewise']]='global', top_k: Optional[int]=1, ignore_index: Optional[int]=None, validate_args: bool=True) -> Tensor:
    """Compute `Specificity`_.

    .. math:: \\text{Specificity} = \\frac{\\text{TN}}{\\text{TN} + \\text{FP}}

    Where :math:`\\text{TN}` and :math:`\\text{FP}` represent the number of true negatives and
    false positives respecitively.

    This function is a simple wrapper to get the task specific versions of this metric, which is done by setting the
    ``task`` argument to either ``'binary'``, ``'multiclass'`` or ``multilabel``. See the documentation of
    :func:`~torchmetrics.functional.classification.binary_specificity`,
    :func:`~torchmetrics.functional.classification.multiclass_specificity` and
    :func:`~torchmetrics.functional.classification.multilabel_specificity` for the specific
    details of each argument influence and examples.

    LegacyExample:
        >>> from torch import tensor
        >>> preds  = tensor([2, 0, 2, 1])
        >>> target = tensor([1, 1, 2, 0])
        >>> specificity(preds, target, task="multiclass", average='macro', num_classes=3)
        tensor(0.6111)
        >>> specificity(preds, target, task="multiclass", average='micro', num_classes=3)
        tensor(0.6250)

    """
    task = ClassificationTask.from_str(task)
    assert multidim_average is not None
    if task == ClassificationTask.BINARY:
        return binary_specificity(preds, target, threshold, multidim_average, ignore_index, validate_args)
    if task == ClassificationTask.MULTICLASS:
        if not isinstance(num_classes, int):
            raise ValueError(f'`num_classes` is expected to be `int` but `{type(num_classes)} was passed.`')
        if not isinstance(top_k, int):
            raise ValueError(f'`top_k` is expected to be `int` but `{type(top_k)} was passed.`')
        return multiclass_specificity(preds, target, num_classes, average, top_k, multidim_average, ignore_index, validate_args)
    if task == ClassificationTask.MULTILABEL:
        if not isinstance(num_labels, int):
            raise ValueError(f'`num_labels` is expected to be `int` but `{type(num_labels)} was passed.`')
        return multilabel_specificity(preds, target, num_labels, threshold, average, multidim_average, ignore_index, validate_args)
    raise ValueError(f'Not handled value: {task}')