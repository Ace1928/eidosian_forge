from typing import Tuple, Union
import torch
from torch import Tensor
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.checks import _check_same_shape
Compute r2 score also known as `R2 Score_Coefficient Determination`_.

    .. math:: R^2 = 1 - \frac{SS_{res}}{SS_{tot}}

    where :math:`SS_{res}=\sum_i (y_i - f(x_i))^2` is the sum of residual squares, and
    :math:`SS_{tot}=\sum_i (y_i - \bar{y})^2` is total sum of squares. Can also calculate
    adjusted r2 score given by

    .. math:: R^2_{adj} = 1 - \frac{(1-R^2)(n-1)}{n-k-1}

    where the parameter :math:`k` (the number of independent regressors) should
    be provided as the ``adjusted`` argument.

    Args:
        preds: estimated labels
        target: ground truth labels
        adjusted: number of independent regressors for calculating adjusted r2 score.
        multioutput: Defines aggregation in the case of multiple output scores. Can be one of the following strings:

            * ``'raw_values'`` returns full set of scores
            * ``'uniform_average'`` scores are uniformly averaged
            * ``'variance_weighted'`` scores are weighted by their individual variances

    Raises:
        ValueError:
            If both ``preds`` and ``targets`` are not ``1D`` or ``2D`` tensors.
        ValueError:
            If ``len(preds)`` is less than ``2`` since at least ``2`` samples are needed to calculate r2 score.
        ValueError:
            If ``multioutput`` is not one of ``raw_values``, ``uniform_average`` or ``variance_weighted``.
        ValueError:
            If ``adjusted`` is not an ``integer`` greater than ``0``.

    Example:
        >>> from torchmetrics.functional.regression import r2_score
        >>> target = torch.tensor([3, -0.5, 2, 7])
        >>> preds = torch.tensor([2.5, 0.0, 2, 8])
        >>> r2_score(preds, target)
        tensor(0.9486)

        >>> target = torch.tensor([[0.5, 1], [-1, 1], [7, -6]])
        >>> preds = torch.tensor([[0, 2], [-1, 2], [8, -5]])
        >>> r2_score(preds, target, multioutput='raw_values')
        tensor([0.9654, 0.9082])

    