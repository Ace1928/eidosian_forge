import warnings
from typing import List, Optional, Tuple, Union
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.functional.classification.precision_recall_curve import (
from torchmetrics.functional.classification.roc import (
from torchmetrics.utilities.enums import ClassificationTask
Compute the highest possible specificity value given the minimum sensitivity thresholds provided.

    This is done by first calculating the Receiver Operating Characteristic (ROC) curve for different thresholds and
    the find the specificity for a given sensitivity level.

    This function is a simple wrapper to get the task specific versions of this metric, which is done by setting the
    ``task`` argument to either ``'binary'``, ``'multiclass'`` or ``multilabel``. See the documentation of
    :func:`~torchmetrics.functional.classification.binary_specificity_at_sensitivity`,
    :func:`~torchmetrics.functional.classification.multiclass_specificity_at_sensitivity` and
    :func:`~torchmetrics.functional.classification.multilabel_specificity_at_sensitivity` for the specific details of
    each argument influence and examples.

    