import itertools
from typing import Optional
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.functional.classification.confusion_matrix import _multiclass_confusion_matrix_update
from torchmetrics.functional.nominal.utils import (
def cramers_v_matrix(matrix: Tensor, bias_correction: bool=True, nan_strategy: Literal['replace', 'drop']='replace', nan_replace_value: Optional[float]=0.0) -> Tensor:
    """Compute `Cramer's V`_ statistic between a set of multiple variables.

    This can serve as a convenient tool to compute Cramer's V statistic for analyses of correlation between categorical
    variables in your dataset.

    Args:
        matrix: A tensor of categorical (nominal) data, where:
            - rows represent a number of data points
            - columns represent a number of categorical (nominal) features
        bias_correction: Indication of whether to use bias correction.
        nan_strategy: Indication of whether to replace or drop ``NaN`` values
        nan_replace_value: Value to replace ``NaN``s when ``nan_strategy = 'replace'``

    Returns:
        Cramer's V statistic for a dataset of categorical variables

    Example:
        >>> from torchmetrics.functional.nominal import cramers_v_matrix
        >>> _ = torch.manual_seed(42)
        >>> matrix = torch.randint(0, 4, (200, 5))
        >>> cramers_v_matrix(matrix)
        tensor([[1.0000, 0.0637, 0.0000, 0.0542, 0.1337],
                [0.0637, 1.0000, 0.0000, 0.0000, 0.0000],
                [0.0000, 0.0000, 1.0000, 0.0000, 0.0649],
                [0.0542, 0.0000, 0.0000, 1.0000, 0.1100],
                [0.1337, 0.0000, 0.0649, 0.1100, 1.0000]])

    """
    _nominal_input_validation(nan_strategy, nan_replace_value)
    num_variables = matrix.shape[1]
    cramers_v_matrix_value = torch.ones(num_variables, num_variables, device=matrix.device)
    for i, j in itertools.combinations(range(num_variables), 2):
        x, y = (matrix[:, i], matrix[:, j])
        num_classes = len(torch.cat([x, y]).unique())
        confmat = _cramers_v_update(x, y, num_classes, nan_strategy, nan_replace_value)
        cramers_v_matrix_value[i, j] = cramers_v_matrix_value[j, i] = _cramers_v_compute(confmat, bias_correction)
    return cramers_v_matrix_value