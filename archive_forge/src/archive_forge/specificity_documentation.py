from typing import Optional
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.functional.classification.stat_scores import (
from torchmetrics.utilities.compute import _adjust_weights_safe_divide, _safe_divide
from torchmetrics.utilities.enums import ClassificationTask
Compute `Specificity`_.

    .. math:: \text{Specificity} = \frac{\text{TN}}{\text{TN} + \text{FP}}

    Where :math:`\text{TN}` and :math:`\text{FP}` represent the number of true negatives and
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

    