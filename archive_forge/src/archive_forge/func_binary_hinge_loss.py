from typing import Optional, Tuple
import torch
from torch import Tensor, tensor
from typing_extensions import Literal
from torchmetrics.functional.classification.confusion_matrix import (
from torchmetrics.utilities.data import to_onehot
from torchmetrics.utilities.enums import ClassificationTaskNoMultilabel
def binary_hinge_loss(preds: Tensor, target: Tensor, squared: bool=False, ignore_index: Optional[int]=None, validate_args: bool=False) -> Tensor:
    """Compute the mean `Hinge loss`_ typically used for Support Vector Machines (SVMs) for binary tasks.

    .. math::
        \\text{Hinge loss} = \\max(0, 1 - y \\times \\hat{y})

    Where :math:`y \\in {-1, 1}` is the target, and :math:`\\hat{y} \\in \\mathbb{R}` is the prediction.

    Accepts the following input tensors:

    - ``preds`` (float tensor): ``(N, ...)``. Preds should be a tensor containing probabilities or logits for each
      observation. If preds has values outside [0,1] range we consider the input to be logits and will auto apply
      sigmoid per element.
    - ``target`` (int tensor): ``(N, ...)``. Target should be a tensor containing ground truth labels, and therefore
      only contain {0,1} values (except if `ignore_index` is specified). The value 1 always encodes the positive class.

    Additional dimension ``...`` will be flattened into the batch dimension.

    Args:
        preds: Tensor with predictions
        target: Tensor with true labels
        squared:
            If True, this will compute the squared hinge loss. Otherwise, computes the regular hinge loss.
        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.

    Example:
        >>> from torch import tensor
        >>> from torchmetrics.functional.classification import binary_hinge_loss
        >>> preds = tensor([0.25, 0.25, 0.55, 0.75, 0.75])
        >>> target = tensor([0, 0, 1, 1, 1])
        >>> binary_hinge_loss(preds, target)
        tensor(0.6900)
        >>> binary_hinge_loss(preds, target, squared=True)
        tensor(0.6905)

    """
    if validate_args:
        _binary_hinge_loss_arg_validation(squared, ignore_index)
        _binary_hinge_loss_tensor_validation(preds, target, ignore_index)
    preds, target = _binary_confusion_matrix_format(preds, target, threshold=0.0, ignore_index=ignore_index, convert_to_labels=False)
    measures, total = _binary_hinge_loss_update(preds, target, squared)
    return _hinge_loss_compute(measures, total)