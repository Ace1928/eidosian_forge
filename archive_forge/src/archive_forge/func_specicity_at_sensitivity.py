import warnings
from typing import List, Optional, Tuple, Union
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.functional.classification.precision_recall_curve import (
from torchmetrics.functional.classification.roc import (
from torchmetrics.utilities.enums import ClassificationTask
def specicity_at_sensitivity(preds: Tensor, target: Tensor, task: Literal['binary', 'multiclass', 'multilabel'], min_sensitivity: float, thresholds: Optional[Union[int, List[float], Tensor]]=None, num_classes: Optional[int]=None, num_labels: Optional[int]=None, ignore_index: Optional[int]=None, validate_args: bool=True) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor], Tuple[List[Tensor], List[Tensor], List[Tensor]]]:
    """Compute the highest possible specificity value given the minimum sensitivity thresholds provided.

    .. warning::
        This function was deprecated in v1.3.0 of Torchmetrics and will be removed in v2.0.0.
        Use `specificity_at_sensitivity` instead.

    """
    warnings.warn('This method has will be removed in 2.0.0. Use `specificity_at_sensitivity` instead.', DeprecationWarning, stacklevel=1)
    return specificity_at_sensitivity(preds=preds, target=target, task=task, min_sensitivity=min_sensitivity, thresholds=thresholds, num_classes=num_classes, num_labels=num_labels, ignore_index=ignore_index, validate_args=validate_args)