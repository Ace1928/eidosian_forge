import itertools
from typing import Optional
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.functional.classification.confusion_matrix import _multiclass_confusion_matrix_update
from torchmetrics.functional.nominal.utils import (
def tschuprows_t(preds: Tensor, target: Tensor, bias_correction: bool=True, nan_strategy: Literal['replace', 'drop']='replace', nan_replace_value: Optional[float]=0.0) -> Tensor:
    """Compute `Tschuprow's T`_ statistic measuring the association between two categorical (nominal) data series.

    .. math::
        T = \\sqrt{\\frac{\\chi^2 / n}{\\sqrt{(r - 1) * (k - 1)}}}

    where

    .. math::
        \\chi^2 = \\sum_{i,j} \\ frac{\\left(n_{ij} - \\frac{n_{i.} n_{.j}}{n}\\right)^2}{\\frac{n_{i.} n_{.j}}{n}}

    where :math:`n_{ij}` denotes the number of times the values :math:`(A_i, B_j)` are observed with :math:`A_i, B_j`
    represent frequencies of values in ``preds`` and ``target``, respectively.

    Tschuprow's T is a symmetric coefficient, i.e. :math:`T(preds, target) = T(target, preds)`.

    The output values lies in [0, 1] with 1 meaning the perfect association.

    Args:
        preds: 1D or 2D tensor of categorical (nominal) data:

            - 1D shape: (batch_size,)
            - 2D shape: (batch_size, num_classes)

        target: 1D or 2D tensor of categorical (nominal) data:

            - 1D shape: (batch_size,)
            - 2D shape: (batch_size, num_classes)

        bias_correction: Indication of whether to use bias correction.
        nan_strategy: Indication of whether to replace or drop ``NaN`` values
        nan_replace_value: Value to replace ``NaN``s when ``nan_strategy = 'replace'``

    Returns:
        Tschuprow's T statistic

    Example:
        >>> from torchmetrics.functional.nominal import tschuprows_t
        >>> _ = torch.manual_seed(42)
        >>> preds = torch.randint(0, 4, (100,))
        >>> target = torch.round(preds + torch.randn(100)).clamp(0, 4)
        >>> tschuprows_t(preds, target)
        tensor(0.4930)

    """
    _nominal_input_validation(nan_strategy, nan_replace_value)
    num_classes = len(torch.cat([preds, target]).unique())
    confmat = _tschuprows_t_update(preds, target, num_classes, nan_strategy, nan_replace_value)
    return _tschuprows_t_compute(confmat, bias_correction)