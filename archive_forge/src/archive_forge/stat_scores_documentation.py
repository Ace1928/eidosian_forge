from typing import List, Optional, Tuple, Union
import torch
from torch import Tensor, tensor
from typing_extensions import Literal
from torchmetrics.utilities.checks import _check_same_shape, _input_format_classification
from torchmetrics.utilities.data import _bincount, select_topk
from torchmetrics.utilities.enums import AverageMethod, ClassificationTask, DataType, MDMCAverageMethod
Compute the number of true positives, false positives, true negatives, false negatives and the support.

    This function is a simple wrapper to get the task specific versions of this metric, which is done by setting the
    ``task`` argument to either ``'binary'``, ``'multiclass'`` or ``multilabel``. See the documentation of
    :func:`~torchmetrics.functional.classification.binary_stat_scores`,
    :func:`~torchmetrics.functional.classification.multiclass_stat_scores` and
    :func:`~torchmetrics.functional.classification.multilabel_stat_scores` for the specific
    details of each argument influence and examples.

    Legacy Example:
        >>> from torch import tensor
        >>> preds  = tensor([1, 0, 2, 1])
        >>> target = tensor([1, 1, 2, 0])
        >>> stat_scores(preds, target, task='multiclass', num_classes=3, average='micro')
        tensor([2, 2, 6, 2, 4])
        >>> stat_scores(preds, target, task='multiclass', num_classes=3, average=None)
        tensor([[0, 1, 2, 1, 1],
                [1, 1, 1, 1, 2],
                [1, 0, 3, 0, 1]])

    