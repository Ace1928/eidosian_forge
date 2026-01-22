from typing import Optional
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.functional.classification.confusion_matrix import (
from torchmetrics.utilities.enums import ClassificationTaskNoMultilabel
def cohen_kappa(preds: Tensor, target: Tensor, task: Literal['binary', 'multiclass'], threshold: float=0.5, num_classes: Optional[int]=None, weights: Optional[Literal['linear', 'quadratic', 'none']]=None, ignore_index: Optional[int]=None, validate_args: bool=True) -> Tensor:
    """Calculate `Cohen's kappa score`_ that measures inter-annotator agreement. It is defined as.

    .. math::
        \\kappa = (p_o - p_e) / (1 - p_e)

    where :math:`p_o` is the empirical probability of agreement and :math:`p_e` is
    the expected agreement when both annotators assign labels randomly. Note that
    :math:`p_e` is estimated using a per-annotator empirical prior over the
    class labels.

    This function is a simple wrapper to get the task specific versions of this metric, which is done by setting the
    ``task`` argument to either ``'binary'`` or ``'multiclass'``. See the documentation of
    :func:`~torchmetrics.functional.classification.binary_cohen_kappa` and
    :func:`~torchmetrics.functional.classification.multiclass_cohen_kappa` for the specific details of
    each argument influence and examples.

    Legacy Example:
        >>> from torch import tensor
        >>> target = tensor([1, 1, 0, 0])
        >>> preds = tensor([0, 1, 0, 0])
        >>> cohen_kappa(preds, target, task="multiclass", num_classes=2)
        tensor(0.5000)

    """
    task = ClassificationTaskNoMultilabel.from_str(task)
    if task == ClassificationTaskNoMultilabel.BINARY:
        return binary_cohen_kappa(preds, target, threshold, weights, ignore_index, validate_args)
    if task == ClassificationTaskNoMultilabel.MULTICLASS:
        if not isinstance(num_classes, int):
            raise ValueError(f'`num_classes` is expected to be `int` but `{type(num_classes)} was passed.`')
        return multiclass_cohen_kappa(preds, target, num_classes, weights, ignore_index, validate_args)
    raise ValueError(f'Not handled value: {task}')