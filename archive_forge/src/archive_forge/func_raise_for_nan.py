from __future__ import annotations
import numpy as np
from pandas._libs import (
def raise_for_nan(value, method: str) -> None:
    if lib.is_float(value) and np.isnan(value):
        raise ValueError(f"Cannot perform logical '{method}' with floating NaN")