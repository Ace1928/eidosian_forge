from typing import Any, List, Optional, Union
import numpy as np
def uniform_to_continuous(value: Any, low: float, high: float, log: bool=False, base: Optional[float]=None) -> Any:
    if low >= high:
        return low if np.isscalar(value) else np.full(np.shape(value))
    if not log:
        return value * (high - low) + low
    if base is None:
        ll, lh = (np.log(low), np.log(high))
        return np.exp(value * (lh - ll) + ll)
    else:
        b = np.log(base)
        ll, lh = (np.log(low) / b, np.log(high) / b)
        return np.power(base, value * (lh - ll) + ll)