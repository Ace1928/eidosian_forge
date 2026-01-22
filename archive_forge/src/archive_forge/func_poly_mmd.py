from typing import Any, List, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor
from torch.nn import Module
from torchmetrics.image.fid import NoTrainInceptionV3
from torchmetrics.metric import Metric
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.data import dim_zero_cat
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE, _TORCH_FIDELITY_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE
def poly_mmd(f_real: Tensor, f_fake: Tensor, degree: int=3, gamma: Optional[float]=None, coef: float=1.0) -> Tensor:
    """Adapted from `KID Score`_."""
    k_11 = poly_kernel(f_real, f_real, degree, gamma, coef)
    k_22 = poly_kernel(f_fake, f_fake, degree, gamma, coef)
    k_12 = poly_kernel(f_real, f_fake, degree, gamma, coef)
    return maximum_mean_discrepancy(k_11, k_12, k_22)