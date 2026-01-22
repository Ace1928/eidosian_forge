import itertools
from typing import Optional
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.functional.classification.confusion_matrix import _multiclass_confusion_matrix_update
from torchmetrics.functional.nominal.utils import (
def theils_u(preds: Tensor, target: Tensor, nan_strategy: Literal['replace', 'drop']='replace', nan_replace_value: Optional[float]=0.0) -> Tensor:
    """Compute `Theils Uncertainty coefficient`_ statistic measuring the association between two nominal data series.

    .. math::
        U(X|Y) = \\frac{H(X) - H(X|Y)}{H(X)}

    where :math:`H(X)` is entropy of variable :math:`X` while :math:`H(X|Y)` is the conditional entropy of :math:`X`
    given :math:`Y`.

    Theils's U is an asymmetric coefficient, i.e. :math:`TheilsU(preds, target) \\neq TheilsU(target, preds)`.

    The output values lies in [0, 1]. 0 means y has no information about x while value 1 means y has complete
    information about x.

    Args:
        preds: 1D or 2D tensor of categorical (nominal) data
            - 1D shape: (batch_size,)
            - 2D shape: (batch_size, num_classes)
        target: 1D or 2D tensor of categorical (nominal) data
            - 1D shape: (batch_size,)
            - 2D shape: (batch_size, num_classes)
        nan_strategy: Indication of whether to replace or drop ``NaN`` values
        nan_replace_value: Value to replace ``NaN``s when ``nan_strategy = 'replace'``

    Returns:
        Tensor containing Theil's U statistic

    Example:
        >>> from torchmetrics.functional.nominal import theils_u
        >>> _ = torch.manual_seed(42)
        >>> preds = torch.randint(10, (10,))
        >>> target = torch.randint(10, (10,))
        >>> theils_u(preds, target)
        tensor(0.8530)

    """
    num_classes = len(torch.cat([preds, target]).unique())
    confmat = _theils_u_update(preds, target, num_classes, nan_strategy, nan_replace_value)
    return _theils_u_compute(confmat)