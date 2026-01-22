from __future__ import annotations
import inspect
from dataclasses import dataclass
from typing import (
import numpy as np
from scipy.stats._common import ConfidenceInterval
from scipy.stats._qmc import check_random_state
from scipy.stats._resampling import BootstrapResult
from scipy.stats import qmc, bootstrap
def indices_method(f_A, f_B, f_AB):
    """Wrap indices method to ensure proper output dimension.

        1D when single output, 2D otherwise.
        """
    return np.squeeze(indices_method_(f_A=f_A, f_B=f_B, f_AB=f_AB))