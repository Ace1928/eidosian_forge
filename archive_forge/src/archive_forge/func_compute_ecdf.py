from typing import Any, Callable, Optional, Tuple
import warnings
import numpy as np
from scipy.stats import uniform, binom
def compute_ecdf(sample: np.ndarray, eval_points: np.ndarray) -> np.ndarray:
    """Compute ECDF of the sorted `sample` at the evaluation points."""
    return np.searchsorted(sample, eval_points, side='right') / len(sample)