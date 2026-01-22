from typing import List, Optional, Tuple, Union
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.functional.classification.precision_recall_curve import (
from torchmetrics.utilities.compute import _safe_divide
from torchmetrics.utilities.data import _bincount
from torchmetrics.utilities.enums import ClassificationTask
from torchmetrics.utilities.prints import rank_zero_warn
def average_precision(preds: Tensor, target: Tensor, task: Literal['binary', 'multiclass', 'multilabel'], thresholds: Optional[Union[int, List[float], Tensor]]=None, num_classes: Optional[int]=None, num_labels: Optional[int]=None, average: Optional[Literal['macro', 'weighted', 'none']]='macro', ignore_index: Optional[int]=None, validate_args: bool=True) -> Optional[Tensor]:
    """Compute the average precision (AP) score.

    The AP score summarizes a precision-recall curve as an weighted mean of precisions at each threshold, with the
    difference in recall from the previous threshold as weight:

    .. math::
        AP = \\sum{n} (R_n - R_{n-1}) P_n

    where :math:`P_n, R_n` is the respective precision and recall at threshold index :math:`n`. This value is
    equivalent to the area under the precision-recall curve (AUPRC).

    This function is a simple wrapper to get the task specific versions of this metric, which is done by setting the
    ``task`` argument to either ``'binary'``, ``'multiclass'`` or ``multilabel``. See the documentation of
    :func:`~torchmetrics.functional.classification.binary_average_precision`,
    :func:`~torchmetrics.functional.classification.multiclass_average_precision` and
    :func:`~torchmetrics.functional.classification.multilabel_average_precision`
    for the specific details of each argument influence and examples.

    Legacy Example:
        >>> from torchmetrics.functional.classification import average_precision
        >>> pred = torch.tensor([0.0, 1.0, 2.0, 3.0])
        >>> target = torch.tensor([0, 1, 1, 1])
        >>> average_precision(pred, target, task="binary")
        tensor(1.)

        >>> pred = torch.tensor([[0.75, 0.05, 0.05, 0.05, 0.05],
        ...                      [0.05, 0.75, 0.05, 0.05, 0.05],
        ...                      [0.05, 0.05, 0.75, 0.05, 0.05],
        ...                      [0.05, 0.05, 0.05, 0.75, 0.05]])
        >>> target = torch.tensor([0, 1, 3, 2])
        >>> average_precision(pred, target, task="multiclass", num_classes=5, average=None)
        tensor([1.0000, 1.0000, 0.2500, 0.2500,    nan])

    """
    task = ClassificationTask.from_str(task)
    if task == ClassificationTask.BINARY:
        return binary_average_precision(preds, target, thresholds, ignore_index, validate_args)
    if task == ClassificationTask.MULTICLASS:
        if not isinstance(num_classes, int):
            raise ValueError(f'`num_classes` is expected to be `int` but `{type(num_classes)} was passed.`')
        return multiclass_average_precision(preds, target, num_classes, average, thresholds, ignore_index, validate_args)
    if task == ClassificationTask.MULTILABEL:
        if not isinstance(num_labels, int):
            raise ValueError(f'`num_labels` is expected to be `int` but `{type(num_labels)} was passed.`')
        return multilabel_average_precision(preds, target, num_labels, average, thresholds, ignore_index, validate_args)
    return None