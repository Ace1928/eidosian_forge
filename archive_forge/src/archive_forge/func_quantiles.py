import numpy as np
def quantiles(self) -> np.ndarray:
    """Returns ndarray with 0, 10, 50, 90, and 100 percentiles."""
    if not self.count:
        return np.ndarray([], dtype=np.float32)
    else:
        return np.nanpercentile(self.items[:self.count], [0, 10, 50, 90, 100]).tolist()